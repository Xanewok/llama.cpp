"""Same-binary fold identity and negative controls. One model process at a time, each bounded to 60 min."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
model = os.environ['BW_MODEL']
bin_path = root / 'build-cpu/bin/bw-np2-driver'
gate = root / 'codex/mseq/gate.sh'
summary = {}


def run(tag, meta, np, kv, k, seed, fold, extra=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', LLAMA_MTP_CHAIN='0', NP2_N_MAX=str(k), NP2_THREADS='4',
               NP2_KV_UNIFIED=str(kv), HOST_KV_Q8='1', GGML_SCHED_INPUT_STAGING='1', LLAMA_MTP_FOLD_C=str(fold),
               LLAMA_MTP_DRAFT_DUMP=str(out / f'{tag}.dump'), LLAMA_ROUND_TRACE=str(out / f'{tag}.trace'))
    env.update(extra or {})
    start = time.monotonic()
    with (out / f'{tag}.log').open('w') as log:
        try:
            rc = subprocess.run(['bash', str(gate), str(bin_path), model, str(out / f'{tag}.json'),
                                 str(meta), str(np), str(seed), '64'], env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=3600).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    elapsed = time.monotonic() - start
    print(f'M: {tag} rc={rc} elapsed_s={elapsed:.1f}', flush=True)
    if rc:
        raise SystemExit(rc)
    data = json.loads((out / f'{tag}.json').read_text())
    dump = (out / f'{tag}.dump').read_text().splitlines()
    phases = [p for line in (out / f'{tag}.trace').read_text().splitlines() for p in json.loads(line)['phases']]
    counts = {}
    for ph in phases:
        assert ph['result'] == 0, (tag, ph)
        for c in ph['counters']:
            counts[c['kind']] = counts.get(c['kind'], 0) + c['count']
    result = dict(data=data, dump=dump, counts=counts, c=sum(p['phase'] == 'C' for p in phases),
                  log=(out / f'{tag}.log').read_text())
    summary[tag] = dict(elapsed_s_M=elapsed, counters=counts, catchup_decodes_M=result['c'], dump_rows_M=len(dump))
    return result


def pair(tag, meta, np, kv, k, seed, extra=None):
    a = run(tag + '-off', meta, np, kv, k, seed, 0, extra)
    b = run(tag + '-on', meta, np, kv, k, seed, 1, extra)
    assert a['data'] == b['data'], (tag, 'tokens/counters differ')
    assert a['dump'] and a['dump'] == b['dump'], (tag, 'draft dump differs')
    assert not a['counts'].get('mtp_fold', 0), (tag, 'off folded')
    assert b['counts'].get('mtp_fold', 0) > 0 and b['c'] < a['c'], (tag, 'no catch-up saved')
    assert f'mtp_fold_decode: engaged n_seq={np}' in b['log'], (tag, 'no joint engagement')
    if np == 2:
        assert {int(row.split()[1]) for row in b['dump']} == {0, 1}, (tag, 'missing slot drafts')
    print('PASS', tag, 'tokens, per-slot counters and draft dumps byte-identical', flush=True)
    return a, b


try:
    refs = {}
    for meta in (0, 1):
        for np in (1, 2):
            for kv in ((0,) if np == 1 else (0, 1)):
                for k in (3, 5):
                    for seed in (1729, 2718):
                        tag = f'm{meta}-np{np}-kv{kv}-k{k}-s{seed}'
                        refs[tag] = pair(tag, meta, np, kv, k, seed)
    base_tag = 'm0-np2-kv0-k3-s1729'
    base = refs[base_tag][0]
    mutant = run('mutant-swap', 0, 2, 0, 3, 1729, 1, {'LLAMA_MTP_FOLD_DEBUG_SWAP_SEQ': '1'})
    assert 'mtp_fold_mutant: wrong sequence engaged' in mutant['log'], 'mutant did not engage'
    assert base['data'] != mutant['data'], 'wrong-sequence mutant did not change outputs'
    forced = run('forced-layout', 0, 2, 0, 3, 1729, 1, {'LLAMA_MTP_FOLD_DEBUG_LAYOUT': '1'})
    assert base['data'] == forced['data'] and base['dump'] == forced['dump'], 'layout fallback differs'
    assert forced['counts'].get('mtp_fold_layout_fallback', 0) > 0 and not forced['counts'].get('mtp_fold', 0)
    assert forced['c'] == base['c'], 'layout fallback did not take separate catch-up'
    # Slot 1 generates without drafting, while slot 0 drafts in the same request batch.
    a = run('mixed-off', 0, 2, 1, 3, 1729, 0, {'NP2_SLOT1_N_MAX': '0'})
    b = run('mixed-on', 0, 2, 1, 3, 1729, 1, {'NP2_SLOT1_N_MAX': '0'})
    assert a['data'] == b['data'] and a['dump'] == b['dump'], 'mixed identity'
    assert {int(row.split()[1]) for row in b['dump']} == {0}, 'mixed control drafted slot 1'
    assert all(ph['stats'][1]['draft'] == 0 for ph in b['data']['phases']), 'mixed slot draft counters'
    assert b['counts'].get('mtp_fold', 0) > 0, 'mixed never folded'
    summary['rc'] = 0
except Exception as exc:
    summary['rc'] = 2
    summary['failure'] = str(exc)
    print('FAIL', repr(exc), flush=True)
finally:
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
sys.exit(summary.get('rc', 2))
