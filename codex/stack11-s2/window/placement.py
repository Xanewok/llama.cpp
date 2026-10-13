#!/usr/bin/env python3
"""Placement proof of pkg 90 on the stack11-s2 builds, at the four served shapes (J5q and IQ4_XS, np1 and np2).

Each (profile, build) boots serve.sh's exact argv/env on that build's llama-server with GGML_SCHED_DEBUG=2 and verbosity 5, serves one greedy
16-token request per slot (np2: two concurrent, different prompts), records nvidia-smi while up, stops. Builds:
  base   frozen/stack11-base: the negative control, token_embd in CPU_Mapped, GET_ROWS on the CPU, a CPU split before the first Meta split
  s2     frozen/stack11-s2:   token_embd in CUDA_Host_Mapped, GET_ROWS on Meta (both GPUs), no CPU split before Meta, tokens equal base
  nok    frozen/stack11-s2-nok: as s2
A void control (base does not show the CPU path) is rc 5, any other failed gate rc 8, Xid 3, busy GPUs/port 4.

  placement.py run PROFILE BUILD ROOT OUT   boot one build, write OUT/PROFILE/BUILD/proof.json     env: PORT XID_SINCE CUDA_VISIBLE_DEVICES
  placement.py judge OUT                    gates every proof.json under OUT, writes OUT/PLACEMENT.md
  placement.py selftest                     analyzer and judge on recorded master logs (fixtures/) and synthetic mutants
"""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import health

HERE = Path(__file__).resolve().parent
PROFILES = {'swift15-j5q': 1, 'swift15-j5q-duo': 2, 'swift15-iq4': 1, 'swift15-iq4-duo': 2}
BUILDS = ('base', 's2', 'nok')
PROMPTS = ('The capital of France is', 'The first three prime numbers are')
# the debug log writes "<ts> D " before every GGML_LOG_DEBUG call, inside a node record too
NOISE = re.compile(r'\d+\.\d{2}\.\d{3}\.\d{3} [A-Z] ')
NODE = re.compile(r'node #\s*(\d+) \(\s*GET_ROWS\): (.{20}) \((.{5})\) \[(.{5}) (.{8})\] use=\d+,c=\d+:(.*)$')
SRC = re.compile(r'(.{20}) \((.{5})\) \[(.{5}) (.{8})\]')
SPLIT = re.compile(r'## SPLIT #(\d+): (\S+) # ')
BUF = re.compile(r'load_tensors: +(\S+) model buffer size = +([\d.]+) MiB')
WANT_BUF = {'base': 'CPU_Mapped', 's2': 'CUDA_Host_Mapped', 'nok': 'CUDA_Host_Mapped'}


def analyze(log):
    """token_embd GET_ROWS node backends, split lists per graph, model buffers."""
    rows, graphs, buffers = [], [], {}
    with open(log, errors='replace') as stream:
        for raw in stream:
            line = NOISE.sub('', raw.rstrip('\n'))
            if 'GET_ROWS' in line and 'node #' in line:
                m = NODE.search(line)
                src = SRC.search(m.group(6)) if m else None
                if src and src.group(1).strip().startswith('token_embd'):
                    rows.append((m.group(4).strip(), m.group(0)[:300]))
            if '## SPLIT #' in line:
                for m in SPLIT.finditer(line):
                    if m.group(1) == '0':
                        graphs.append([])
                    if graphs:
                        graphs[-1].append(m.group(2))
            m = BUF.search(line)
            if m:
                buffers[m.group(1)] = float(m.group(2))
    meta = [g for g in graphs if any(b.startswith('Meta') for b in g)]
    cpu_before_meta = [g for g in meta if any(b == 'CPU' for b in g[:next(i for i, b in enumerate(g) if b.startswith('Meta'))])]
    backends = sorted({b for b, _ in rows})
    return {'token_embd_get_rows': len(rows), 'get_rows_backends': backends,
            'get_rows_on_meta': bool(rows) and all(b.startswith('Meta') for b in backends),
            'get_rows_on_cpu': any(b == 'CPU' for b in backends),
            'graphs': len(graphs), 'meta_graphs': len(meta), 'cpu_before_meta_graphs': len(cpu_before_meta),
            'both_gpus': sorted({g for gr in meta for g in gr if g.startswith('Meta(CUDA0,CUDA1)')}),
            'buffers': buffers}


def completion(port, prompt, out):
    body = json.dumps({'prompt': prompt, 'n_predict': 16, 'temperature': 0, 'seed': 1}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', body, {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=300) as reply:
            out.update(json.loads(reply.read()))
        return None if out.get('tokens_predicted') else 'no tokens'
    except (OSError, ValueError) as e:
        return f'{type(e).__name__}: {e}'


def run(profile, build, root, out_dir):
    out = Path(out_dir) / profile / build
    out.mkdir(parents=True, exist_ok=True)
    port = int(os.environ['PORT'])
    since = os.environ.get('XID_SINCE') or time.strftime('%Y-%m-%d %H:%M:%S')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    bin_dir = Path(root) / 'build/bin'
    spec = health.launch(profile, port)
    if set(spec['env']['CUDA_VISIBLE_DEVICES'].split(',')) != set(visible.split(',')):
        raise health.Fail(6, 'serve.sh devices are not the window GPUs')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_'))}
    env.update(spec['env'])
    env.update(LLAMA_ARG_LOG_VERBOSITY='5', GGML_SCHED_DEBUG='2', LD_LIBRARY_PATH=str(bin_dir))
    argv = [str(bin_dir / 'llama-server'), *spec['args']]
    (out / 'launch.json').write_text(json.dumps(dict(argv=argv, env={k: v for k, v in env.items() if k.startswith(('GGML_', 'LLAMA_'))}), indent=2))
    if health.xid(since):
        raise health.Fail(3, 'Xid before the boot')
    used = health.gpu_used()
    if used is None or max(used) > 1500:
        raise health.Fail(4, f'GPUs hold {used} MiB before the boot')
    if health.port_busy(port):
        raise health.Fail(4, f'port {port} busy')
    log = out / 'server.log'
    with log.open('w') as stream:
        p = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    birth = health.start_time(p.pid)
    failure, answers, smi = None, [{} for _ in range(spec['np'])], ''
    try:
        t0 = time.monotonic()
        while not health.healthy(port):
            if p.poll() is not None:
                raise RuntimeError(f'server exited rc={p.returncode} before /health')
            if time.monotonic() - t0 > health.BOOT_S:
                raise RuntimeError(f'not healthy within {health.BOOT_S} s')
            time.sleep(1)
        errors = [None] * spec['np']
        threads = [threading.Thread(target=lambda i=i: errors.__setitem__(i, completion(port, PROMPTS[i], answers[i]))) for i in range(spec['np'])]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        smi = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=60).stdout
        if any(errors):
            raise RuntimeError(f'request failed: {[e for e in errors if e]}')
    except RuntimeError as e:
        failure = str(e)
    finally:
        stopped = health.stop(p, birth)
    time.sleep(3)
    gpus = [dict(zip(('index', 'uuid', 'used_MiB', 'free_MiB', 'total_MiB'), (f.strip() for f in row.split(',')))) for row in smi.splitlines() if row.strip()]
    result = analyze(log) | {'profile': profile, 'build': build, 'np': spec['np'], 'failure': failure, 'stop': stopped,
                             'contents': [a.get('content') for a in answers], 'gpus': [g for g in gpus if g.get('uuid') in visible.split(',')]}
    (out / 'proof.json').write_text(json.dumps(result, indent=1))
    print(f'placement {profile} {build}: buffers={result["buffers"]} get_rows={result["get_rows_backends"]} meta={result["meta_graphs"]}/{result["graphs"]} '
          f'cpu_before_meta={result["cpu_before_meta_graphs"]} failure={failure}')
    if health.xid(since):
        raise health.Fail(3, 'Xid during the boot')
    return 8 if failure else 0


def gates(r, profile):
    """[(build, gate, ok, void)] for one profile; r maps build -> proof."""
    out = []
    base = r.get('base')
    for build in BUILDS:
        x = r.get(build)
        if x is None:
            out.append((build, 'ran', False, False))
            continue
        out.append((build, 'no failure', not x['failure'], False))
        out.append((build, f'token_embd buffer {WANT_BUF[build]} (644 MiB class)', x['buffers'].get(WANT_BUF[build], 0) > 100 and
                    (build != 'base') == ('CUDA_Host_Mapped' in x['buffers']), False))
        out.append((build, 'Meta(CUDA0,CUDA1) graphs seen, token_embd GET_ROWS nodes seen', x['meta_graphs'] > 0 and bool(x['both_gpus']) and x['token_embd_get_rows'] > 0, False))
        if build == 'base':
            # the control: without a CPU lookup and a CPU split here the detector proves nothing for s2
            out.append((build, 'token_embd GET_ROWS on the CPU (control)', x['get_rows_on_cpu'], True))
            out.append((build, 'CPU split before Meta (control)', x['cpu_before_meta_graphs'] > 0, True))
        else:
            out.append((build, 'every token_embd GET_ROWS on Meta (both GPUs), none on the CPU', x['get_rows_on_meta'] and not x['get_rows_on_cpu'], False))
            out.append((build, 'no CPU split before the first Meta split', x['cpu_before_meta_graphs'] == 0, False))
            out.append((build, 'tokens equal base', bool(base) and bool(x['contents']) and all(x['contents']) and x['contents'] == base['contents'], False))
    return out


def judge(out_dir):
    out = Path(out_dir)
    lines, ok_all, void = ['# pkg 90 placement proof on stack11-s2 (GGML_SCHED_DEBUG=2, one greedy 16-token request per slot)', ''], True, False
    for profile in PROFILES:
        r = {b: json.loads((out / profile / b / 'proof.json').read_text()) for b in BUILDS if (out / profile / b / 'proof.json').exists()}
        lines.append(f'## {profile} (np={PROFILES[profile]})')
        for build, name, ok, is_control in gates(r, profile):
            lines.append(f'{"PASS" if ok else "FAIL"}  {build}: {name}')
            if not ok:
                ok_all = False
                void = void or is_control
        for build, x in r.items():
            lines.append(f'    {build}: buffers={x["buffers"]} get_rows={x["get_rows_backends"]} graphs={x["graphs"]} meta={x["meta_graphs"]} '
                         f'cpu_before_meta={x["cpu_before_meta_graphs"]} content={x["contents"]!r}')
            lines += [f'    {build} gpu {g["index"]} used={g["used_MiB"]} free={g["free_MiB"]} total={g["total_MiB"]} MiB (M)' for g in x['gpus']]
        lines.append('')
    lines.append(f'verdict: {"PASS" if ok_all else "FAIL"}')
    (out / 'PLACEMENT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if ok_all else 5 if void else 8


def selftest():
    base, zc = analyze(HERE / 'fixtures/placement-base.log'), analyze(HERE / 'fixtures/placement-zc.log')
    assert base['get_rows_on_cpu'] and not base['get_rows_on_meta'] and base['buffers'].get('CPU_Mapped', 0) > 100, base
    assert zc['get_rows_on_meta'] and not zc['get_rows_on_cpu'] and zc['buffers'].get('CUDA_Host_Mapped', 0) > 100, zc
    assert zc['cpu_before_meta_graphs'] == 0 and zc['meta_graphs'] > 0 and zc['both_gpus'], zc

    def proof(a, build, contents):
        return a | {'build': build, 'failure': None, 'contents': contents, 'gpus': []}
    r = {'base': proof(base, 'base', ['a']), 's2': proof(zc, 's2', ['a']), 'nok': proof(zc, 'nok', ['a'])}
    base_ctl = dict(base, cpu_before_meta_graphs=max(base['cpu_before_meta_graphs'], 1), meta_graphs=max(base['meta_graphs'], 1), both_gpus=zc['both_gpus'])
    r['base'] = proof(base_ctl, 'base', ['a'])
    assert all(ok for _, _, ok, _ in gates(r, 'swift15-j5q')), [g for g in gates(r, 'swift15-j5q') if not g[2]]
    # mutants: tokens differ; s2 shows the CPU path; base shows no CPU path (void control); mapped buffer missing
    assert not all(ok for *_, ok, _ in [(b, n, o, v) for b, n, o, v in gates(dict(r, s2=proof(zc, 's2', ['b'])), 'p')])
    assert any(not ok for _, _, ok, _ in gates(dict(r, s2=proof(base_ctl, 's2', ['a'])), 'p'))
    void = [g for g in gates(dict(r, base=proof(zc | {'buffers': {'CPU_Mapped': 644.0}}, 'base', ['a'])), 'p') if not g[2]]
    assert void and any(v for *_, v in void), void
    assert any(not ok for _, _, ok, _ in gates(dict(r, nok=proof(zc | {'buffers': {'CPU_Mapped': 644.0}}, 'nok', ['a'])), 'p'))
    assert any(not ok for _, _, ok, _ in gates({'base': r['base'], 's2': r['s2']}, 'p'))
    print('placement: analyzer on recorded master logs, judge on synthetic mutants PASS')


def main():
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
        return 0
    if len(a) == 2 and a[0] == 'judge':
        return judge(a[1])
    if len(a) == 5 and a[0] == 'run' and a[1] in PROFILES and a[2] in BUILDS:
        try:
            return run(a[1], a[2], a[3], a[4])
        except health.Fail as e:
            print(f'placement {a[1]} {a[2]}: FAIL rc={e.code} {e.reason}', file=sys.stderr)
            return e.code
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main())
