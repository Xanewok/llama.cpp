"""CPU-only contract, capture and execution-refusal checks for labelled dry manifests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {}
CLEAN = {'HOME': str(Path.home()), 'PATH': '/usr/local/bin:/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '',
         'GIT_NO_LAZY_FETCH': '1', 'PYTHONDONTWRITEBYTECODE': '1'}


def check(name, cmd, env=None):
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    RESULTS[name] = {'cmd': cmd, 'rc': proc.returncode, 'output': proc.stdout + proc.stderr}
    assert proc.returncode == 0, (name, proc.stdout, proc.stderr)


def main():
    os.chdir(ROOT)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    check('shell-syntax', ['bash', '-n', 'codex/bw-vanilla.sh', 'codex/gpu_ab.sh'])
    check('shlint', [sys.executable, '-m', 'gpuq.shlint', str(ROOT/'codex/bw-vanilla.sh'), str(ROOT/'codex/gpu_ab.sh')],
          dict(os.environ, PYTHONPATH='/home/xanewok/repos/gpuq'))
    check('window-preflight', ['bash', 'codex/bw-vanilla.sh', '--preflight'])
    check('window-contract', ['gpuq', 'submit', '--check', 'codex/bw-vanilla.toml'])
    check('window-clean-contract', ['/home/xanewok/repos/gpuq/bin/gpuq', 'submit', '--check', str(ROOT/'codex/bw-vanilla.toml')], CLEAN)
    check('window-clean-preflight', ['bash', str(ROOT/'codex/bw-vanilla.sh'), '--preflight'], CLEAN)
    total = 0
    for path in sorted((ROOT/'codex/evidence/dry-render').glob('*.gpuq.toml')):
        check(path.stem+'-clean-contract', ['/home/xanewok/repos/gpuq/bin/gpuq', 'submit', '--check', str(path)], CLEAN)
        item = tomllib.loads(path.read_text())
        total += item['budget_s']
        cmd = item['preflight']['cmd']
        check(path.stem+'-preflight', cmd)
        entry = cmd[:cmd.index('--preflight')]
        out = ROOT/'codex/evidence/capture'/path.stem
        check(path.stem+'-capture', entry+['--capture', '--port', '18199', '--out', str(out)])
        for dest in out.iterdir():
            if not dest.is_dir(): continue
            argv = json.loads((dest/'command.json').read_text())
            env = json.loads((dest/'env.json').read_text())
            assert Path(argv[0]).name == 'round-trace'
            assert argv[argv.index('--spec-type')+1] == 'draft-mtp'
            assert argv[argv.index('--spec-draft-sampling')+1] == 'probabilistic'
            assert argv[argv.index('--cache-ram')+1] == '0'
            assert not any('{' in value for value in argv)
            vanilla = any(arm in dest.name for arm in ('vanilla', 'short', 'long'))
            assert ('--spec-draft-vocab' in argv) != vanilla
            assert env.get('LLAMA_ROUND_TRACE') is None
            if vanilla:
                assert all(env.get(k) is None for k in ('GGML_SCHED_INPUT_STAGING', 'LLAMA_ARG_KV_PAD', 'LLAMA_MTP_CHAIN', 'LLAMA_MTP_FOLD_C', 'GGML_CUDA_Q8_DECODE'))
            else:
                assert env['GGML_SCHED_INPUT_STAGING'] == '1' and env['LLAMA_ARG_KV_PAD'] == '4096'
                assert env['GGML_CUDA_Q8_DECODE'] == ('0' if 'iq4' in path.name else '1')
        proc = subprocess.run(entry+['--canary', '--out', str(ROOT/'codex/evidence/refused-canary')],
                              env=dict(os.environ, GPUQ_ACTIVE='1'), capture_output=True, text=True)
        assert proc.returncode == 6 and 'DRY STAND-IN' in proc.stderr
        RESULTS[path.stem+'-dry-refusal'] = {'rc': proc.returncode, 'output': proc.stderr}
    assert total == 10800
    (ROOT/'codex/evidence/validation.json').write_text(json.dumps(RESULTS, indent=2)+'\n')
    print('All window, clean-env, dry-manifest, preflight, capture and dry-refusal checks PASS; budget_s_D=10800')


if __name__ == '__main__':
    main()
