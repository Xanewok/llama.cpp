"""Render from main harness.mkitem, then add the local untraced entry and queue priority; never submit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parent.parent
W = ROOT.parent.parent
EVAL = Path('/home/xanewok/repos/qwen38-eval')


def main():
    summary = Path(sys.argv[1]).resolve()
    assert not summary.is_relative_to(ROOT)
    assert __import__('re').fullmatch(r'stack7-a[0-9]+', summary.parent.name) and summary.name == 'summary.json'
    if 'BW_OUT_ROOT' in os.environ:
        assert summary.parent.parent == Path(os.environ['BW_OUT_ROOT']).resolve()
    sys.path.insert(0, str(EVAL))
    from harness import mkitem
    evidence = json.loads(summary.read_text())
    assert evidence['stage1'] == evidence['stage2'] == 'PASS'
    sha = (W / 'frozen/stack7/SOURCE').read_text().splitlines()[0]
    assert sha == evidence['commit']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1')
    for name in ('stack7-depth-j5q', 'stack7-np2-real-j5q', 'stack7-duo-soak-j5q'):
        experiment = ROOT / f'codex/items/{name}.toml'
        item = ROOT / f'codex/items/{name}.gpuq.toml'
        text = experiment.with_suffix('.toml.in').read_text().replace('@STACK7_SHA@', sha).replace('@SUMMARY@', str(summary))
        assert '@STACK7_SHA@' not in text and '@SUMMARY@' not in text
        experiment.write_text(text)
        command = [sys.executable, '-m', 'harness.mkitem', str(experiment), '--out', str(item)]
        subprocess.run(command, cwd=EVAL, env=env, check=True)
        subprocess.run([sys.executable, "-m", "harness.mkitem", str(experiment), "--check", str(item)], cwd=EVAL, env=env, check=True)
        generated = item.read_text()
        generated = mkitem.wrap_stages(generated, ROOT / 'codex/gpu_ab.sh')
        item.write_text(generated)
        parsed = tomllib.loads(generated)
        assert str(summary) in parsed['pins']['files']
        assert str(W / 'frozen/stack7/build/bin/round-trace') in parsed['pins']['files']
        assert parsed['canary']['timeout_s'] <= 600
        assert all(str(ROOT / f'codex/{p}') in parsed['pins']['files'] for p in ('gpu_ab.sh', 'gpu_stack7.py'))
        result = subprocess.run(['gpuq', 'submit', '--check', str(item)], env=env, capture_output=True, text=True)
        log = result.stdout + result.stderr
        (summary.parent / f'{name}-contract.log').write_text(log)
        print(log, end='')
        assert result.returncode == 0 and 'contract violated' not in log.lower(), (name, result.returncode, log)
        print(f'checked, NOT submitted: {item}')
    evidence['items'] = 'all three contract checks PASS; NOT submitted'
    summary.write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    main()
