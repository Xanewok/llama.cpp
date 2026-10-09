"""Render one diagnostic item; never submit or execute it."""
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVAL = Path('/home/xanewok/repos/qwen38-eval')
SOURCE = Path('/home/xanewok/repos/qwen38-eval-worktrees/wt/stack10b/codex/stack10/rendered/s10-depth-iq4.toml')
FROZEN = Path('/home/xanewok/repos/qwen38-eval-worktrees/frozen/stack10')
RESULTS = EVAL / 'results/2026-10-09-s10-depth-iq4/full-a3'
CCX = ROOT.parent.parent / 'work/aa-load/ccx01.sh'
# 32768: the full single-switch bisect; 90112: confirm the 32k culprit only (one switch, 90k diverged from round 1)
DEPTH = int(os.environ.get('S10_ID_DEPTH', '32768'))
TAG = {32768: '32k', 90112: '90k'}[DEPTH]
OFF = {
    'k6a_off': ('GGML_CUDA_Q8_REUSE', '0'),
    'k11b_off': ('GGML_META_INPUT_STAGE', '0'),
    'k11a_off': ('LLAMA_SYNC_ELIDE', '0'),
    'k10_off': ('GGML_CUDA_TOPK_BATCHED', '0'),
    'k4_off': ('GGML_SSM_CONV_RS_OFF', '1'),
    'k7_off': ('GGML_CUDA_RMS_NORM_SCALE_PAIR', '0'),
    'alpha_off': ('GGML_CUDA_ALPHA_GATE', '0'),
    'rot_off': ('LLAMA_KV_ROT_PERSIST', '0'),
    'q8decode_off': ('GGML_CUDA_Q8_DECODE', '0'),
    'k8_off': ('GGML_CUDA_MMQ_TILES', 'default'),
    'k6b_off': ('GGML_CUDA_ADD_RMS_NORM', '0'),
    'k2_off': ('GGML_CUDA_FATTN_KV_MAX_SHARE', '0'),
}


def section(name, values):
    return '\n'.join([f'[{name}]', *(f'{k} = {json.dumps(v)}' for k, v in values.items()), '']) + '\n'


def main():
    old = tomllib.loads(SOURCE.read_text())
    sys.path.insert(0, str(HERE.parent))
    import s10items
    import engagement
    sys.path.insert(0, str(EVAL))
    from harness import mkitem
    from render_stack10_items import wrap_ccx

    plan = ['stock', *OFF, 'stock'] if DEPTH == 32768 else ['stock', 'q8decode_off', 'stock']
    wrapper = HERE / 'gpu_ab.sh'
    own = [p for directory in (HERE, HERE / 'render_support') for p in directory.iterdir() if p.is_file() and p.suffix in ('.py', '.sh')]
    inherited = s10items.extra_pins(ROOT, 's10-depth-iq4.toml.in')
    top = {k: old[k] for k in ('priority', 'kind', 'devices', 'protected')}
    top |= {'id': f's10-identity-{TAG}', 'title': f'stack10 np1 {TAG} identity: single-switch ablations against recorded stack9-p55',
            'plan': plan, 'out': str(HERE / ('results' if DEPTH == 32768 else f'results-{TAG}')),
            'extra_pins': [*inherited, *map(str, own), str(SOURCE),
                           str(RESULTS / f'0-s9p/depth-{DEPTH}.json'), str(RESULTS / f'1-s10/depth-{DEPTH}.json'),
                           str(RESULTS / 'pins.json'), str(RESULTS / '0-s9p/command.json'), str(RESULTS / '0-s9p/env.json')]}
    text = '\n'.join(f'{k} = {json.dumps(v)}' for k, v in top.items()) + '\n\n'
    for name in ('model', 'draft_vocab'):
        text += section(name, old[name])
    sha = (FROZEN / 'SOURCE').read_text().splitlines()[0]
    patterns = s10items.WAVE1['np1'][:2] + engagement.canary_patterns(['base', 'k7', 'k4', 'k6a', 'k10', 'k11b'], 'np1')
    for name in dict.fromkeys(plan):
        arm = {k: old['arms']['s10'][k] for k in ('bin', 'exe', 'flags')}
        arm |= {'label': name, 'commit': sha, 'root': str(FROZEN), 'repo': old['arms']['s10']['repo'],
                'canary_log': patterns if name == 'stock' else []}
        text += section('arms.' + name, arm)
        env = dict(old['arms']['s10']['env'])
        if name in OFF:
            key, value = OFF[name]
            assert key in s10items.SWITCH_ENVS
            env[key] = value
        text += section('arms.' + name + '.env', env)
    text += section('server', old['server'] | {'knobs': list(dict.fromkeys(old['server']['knobs'] + ['GGML_META_SUBGRAPH_MUTANT']))})
    d = {k: v for k, v in old['depth'].items() if not isinstance(v, dict)}
    d |= {'depths': [8192, DEPTH], 'identity': 'within-arm', 'wall_s': 130 if DEPTH == 32768 else 400}
    text += section('depth', d)
    text += section('depth.prompt', old['depth']['prompt'])
    text += section('depth.request', old['depth']['request'])
    text += section('canary', {'plan': ['stock'], 'depths': [8192], 'proofs': []})
    text += section('canary.env', {'LLAMA_ARG_LOG_VERBOSITY': '5', 'GGML_SSM_CONV_RS_LOG': '1', 'GGML_CUDA_TOPK_LOG': '1',
                                   'GGML_CUDA_MMQ_TILES_LOG': '1'})
    text += '[budget]\ncanary_s = 180\nmargin = 1.3\ncleanup_s = 60\nest_s = 1200\nstartup_grace_s = 90\npreflight_s = 600\n\n'
    text += section('outputs', {'run': 'full-a{attempt}', 'canary': 'canary-a{attempt}', 'preflight': 'preflight-plan'})
    text += section('cpu_evidence', old['cpu_evidence'])
    text += section('diagnostic', {'depth': DEPTH, 'reference': str(RESULTS / f'0-s9p/depth-{DEPTH}.json'),
                                   'archived_stock': str(RESULTS / f'1-s10/depth-{DEPTH}.json')})
    experiment = HERE / f's10-identity-{TAG}.toml'
    experiment.write_text(text.rstrip() + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1',
               S10_IDENTITY_SNAPSHOTS=str(HERE / 'snapshots'), PYTHONPATH=str(HERE / 'render_support') + ':' + str(EVAL))
    raw = HERE / f's10-identity-{TAG}.raw.toml'
    subprocess.run([sys.executable, '-m', 'harness.mkitem', str(experiment), '--out', str(raw)], cwd=EVAL, env=env, check=True)
    rendered = mkitem.wrap_stages(raw.read_text(), wrapper)
    raw.unlink()
    rendered = wrap_ccx(rendered).replace('kind = "measure"', 'host_sensitive = false\nkind = "measure"', 1)
    parsed = tomllib.loads(rendered)
    print('budget_s', parsed['budget_s'])
    assert parsed['budget_s'] <= 45 * 60
    assert len(plan) == (14 if DEPTH == 32768 else 3) and all(a['root'] == str(FROZEN) for a in tomllib.loads(text)['arms'].values())
    item = HERE / f's10-identity-{TAG}.gpuq.toml'
    item.write_text(rendered)
    result = subprocess.run(['gpuq', 'submit', '--check', str(item)], env=env, capture_output=True, text=True)
    (HERE / f'contract-{TAG}.log').write_text(result.stdout + result.stderr)
    print(result.stdout + result.stderr, end='')
    result.check_returncode()
    print(f'Checked only: {item}')


if __name__ == '__main__':
    main()
