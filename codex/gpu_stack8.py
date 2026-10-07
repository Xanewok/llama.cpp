"""Use the main harness with untraced canaries, a boot budget, and scoped kernel-fault gates."""
import copy
import json
import os
from pathlib import Path
import re
import signal
import sys

EVAL = Path(os.environ.get('S8_EVAL', '/home/xanewok/repos/qwen38-eval'))
FIXTURES = EVAL / 'orchestration/gpuq/fixtures'
REAL = EVAL / 'results/2026-10-05-stack6-depth-j5q/full-a1/1-s6'
REAL_NP2 = EVAL / 'results/2026-10-07-gdn-getter-gate-np2-busy-j5q/canary-a1/0-up'
GETTER = EVAL / 'results/2026-10-07-gdn-getter-gate-np1-j5q/canary-a1/0-patch/server.log'
CHAIN = EVAL / 'results/2026-10-06-draft-chain-m1-depth-j5q/canary-a2/0-chain/server.log'
sys.path.insert(0, os.environ['HARNESS_CODE_ROOT'])
from harness import lib, depth, campaign
from harness.__main__ import main as harness_main


def fixture_checks():
    depth.fixture_test(FIXTURES)
    record = json.loads((REAL / 'depth-8192.json').read_text())
    identity = lib.token_identity(record)
    assert len(identity[0]) >= 256
    lib.steady_intervals(record, 40, 5)
    results = {}
    depth.settle(results, 'cross-arm', 's6', 8192, identity)
    changed = copy.deepcopy(record)
    next(c for c in changed['chunks'] if c.get('tokens'))['tokens'][0] += 1
    try:
        depth.settle(results, 'cross-arm', 's7', 8192, lib.token_identity(changed))
    except lib.Fail as e:
        assert e.cls == 'integrity'
    else:
        raise AssertionError('cross-arm token mutant escaped')
    np2 = json.loads((REAL_NP2 / 'depth-8192.json').read_text())
    window = depth.overlap(np2['streams'], 0.8)
    for r in np2['streams']:
        lib.steady_intervals(r, 40, 5, tuple(window['window_ns']))
        lib.token_identity(r)
    lib.verify_log_patterns(GETTER.read_text(), [r'gdn_state_indexed: device=0 graph=\d+ tokens=6 gathers_skipped=[1-9]\d*'], 'recorded getter')
    lib.verify_log_patterns(CHAIN.read_text(), [r'MTP chain: k=5\b'], 'recorded chain')
    candidate = EVAL / 'results/2026-10-07-cand-neginf-depth-j5q/canary-a1/1-cn/server.log'
    candidate_text = candidate.read_text()
    candidate_patterns = [r'set_logits: using backend candidates: n_cand = \d+, n_ban = [1-9]\d*, top_k = 20\b',
                          r'(?s)^(?!.*set_sampler: [^\n]*using CPU)']
    lib.verify_log_patterns(candidate_text, candidate_patterns, 'recorded banned candidates')
    for bad in ((REAL / 'server.log').read_text(), re.sub(r'n_ban = [1-9]\d*', 'n_ban = 0', candidate_text),
                candidate_text + '\nset_sampler: seq_id = 0: using CPU\n'):
        try:
            lib.verify_log_patterns(bad, candidate_patterns, 'candidate proof control')
        except lib.Fail:
            pass
        else:
            raise AssertionError('candidate proof control escaped')
    evidence = Path(os.environ.get('S8_MAC_EVIDENCE', str(Path(__file__).resolve().parent / 'mac-evidence')))
    fallback_patterns = [r'MTP chain disabled: needs one sequence \(n_seq=2\)', r'(?s)^(?!.*MTP chain: k=)']
    lib.verify_log_patterns((evidence / 'np2-on.log').read_text(), fallback_patterns, 'recorded np2 fallback')
    for bad in ((REAL / 'server.log').read_text(), (evidence / 'np2-old-on.log').read_text()):
        try:
            lib.verify_log_patterns(bad, fallback_patterns, 'np2 fallback control')
        except lib.Fail:
            pass
        else:
            raise AssertionError('np2 fallback control escaped')
    # The new fold and projection markers have no GPU artifact until this window builds them. The parser must reject
    # their absence in a real log; their positive engagement gate runs in the queued canary.
    for marker in (r'mtp_fold_decode: engaged\b', r'ssm_out_2d: seqs=2 rows=[1-9]\d* matrices=1\b'):
        try:
            lib.verify_log_patterns((REAL / 'server.log').read_text(), [marker], 'absent new marker')
        except lib.Fail:
            pass
        else:
            raise AssertionError('missing engagement marker escaped')
    x = {'server': {'args': [], 'knobs': ['LLAMA_ROUND_TRACE']}, 'canary': {}, 'depth': {}}
    arm = {'env': {}}
    extra = {'LLAMA_ROUND_TRACE': '/should/not/be/written'}
    env = untraced_env({}, arm, ['LLAMA_ROUND_TRACE'], extra)
    assert 'LLAMA_ROUND_TRACE' not in env and not extra
    print('recorded SSE, native tokens, overlap, memory, kernel journal, getter/chain/candidate/fallback logs and negative controls PASS')
    return 0


original_env = lib.server_env

def untraced_env(base, arm, knobs, extra=None):
    # campaign.canary_arm also checks its extra dict against the resulting environment.
    if extra is not None:
        extra.pop('LLAMA_ROUND_TRACE', None)
    env = original_env(base, arm, knobs, extra)
    env.pop('LLAMA_ROUND_TRACE', None)
    return env


def install_runtime_guards(argv):
    lib.server_env = untraced_env
    campaign.harness_cmd = lambda *args: [sys.executable, '-P', str(Path(__file__).resolve()), *map(str, args)]
    original_start, original_stop = lib.Server.start, lib.Server.stop
    x = lib.read_experiment(argv[1])
    cap = 520 if '--canary' in argv else (x['depth']['wall_s'] - 60 if x['kind'] == 'depth' else 0)

    def expire(signum, frame):
        raise lib.Fail('budget: boot wall expired', 'timeout')

    def start(server):
        pid = original_start(server)
        if cap:
            signal.signal(signal.SIGALRM, expire)
            signal.setitimer(signal.ITIMER_REAL, cap)
        return pid

    def stop(server, grace=30):
        signal.setitimer(signal.ITIMER_REAL, 0)
        return original_stop(server, min(grace, 30))

    original_journal = lib.record_journal
    def journal(since, path, journal_file=None):
        n = original_journal(since, path, journal_file)
        if n is None:
            raise lib.Fail('kernel journal unreadable', 'integrity')
        if n:
            raise lib.Fail('Xid since boot start', 'hazard')
        return n

    lib.Server.start, lib.Server.stop = start, stop
    lib.record_journal = journal


if __name__ == '__main__':
    if sys.argv[1:] == ['--fixtures']:
        sys.exit(fixture_checks())
    install_runtime_guards(sys.argv[1:])
    sys.exit(harness_main())
