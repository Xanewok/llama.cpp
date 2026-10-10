"""Render new manifests in this tree; validation never submits a job."""
import copy
import json
from pathlib import Path
import sys
import tomllib

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[2]
WT = TREE.parents[1]
EVAL = Path('/home/xanewok/repos/qwen38-eval')
CCX = WT / 'work/aa-load/ccx01.sh'
SNAPSHOT = HERE.parent / 'identity/snapshots/harness-385c353bf0ab87df00dc6b9773c80ae4827f0dec25cf821cb88c76e57d3393a9'
sys.path.insert(0, str(SNAPSHOT))
from harness import mkitem
import kld_item


def table(name, values):
    def value(v):
        if isinstance(v, dict):
            return '{ ' + ', '.join(k + ' = ' + value(x) for k, x in v.items()) + ' }'
        return json.dumps(v)
    return f'[{name}]\n' + ''.join(f'{k} = {value(v)}\n' for k, v in values.items()) + '\n'


def experiment_text(x):
    text = ''.join(f'{k} = {json.dumps(v)}\n' for k, v in x.items() if not isinstance(v, dict)) + '\n'
    def emit(name, data):
        result = table(name, {k: v for k, v in data.items() if not isinstance(v, dict)})
        for k, v in data.items():
            if isinstance(v, dict):
                result += emit(name + '.' + k, v)
        return result
    return text + ''.join(emit(k, v) for k, v in x.items() if isinstance(v, dict))


def frozen_pins(name):
    root = WT / 'frozen' / name
    return [str(root / 'SOURCE'), str(root / 'RECIPE.txt'), *map(str, sorted((root / 'build/bin').iterdir()))]


def common_pins():
    own = [str(p) for p in HERE.iterdir() if p.suffix in ('.py', '.sh', '.cpp') or p.name == 'CMakeLists.txt']
    return own + [str(CCX), str(WT / 'bin/buildgate'), *map(str, sorted((SNAPSHOT / 'harness').glob('*.py')))]


def kld_manifest():
    own = common_pins()
    own += [str(kld_item.CORPUS), str(kld_item.MODEL), str(kld_item.REAL_KLF),
            str(EVAL / 'results/2026-10-09-s10-depth-iq4/canary-a3/0-s10/server.log'),
            str(kld_item.FIXTURES / 'xid-lines.txt'), str(kld_item.FIXTURES / 'nvidia-smi-query.csv')]
    own += [str(p) for d in ('include', 'common', 'ggml/include') for p in (TREE / d).glob('*.h')]
    own += [str(TREE / 'src/llama-ext.h')]
    own += frozen_pins('stack9-p55') + frozen_pins('stack10')
    prefix = ['bash', str(CCX), 'bash', str(HERE / 'gpu_ab.sh')]
    out = '/mnt/scratch/work/s10-identity2/s10-kld-iq4/a{attempt}'
    text = '\n'.join(['id = "s10-kld-iq4"', 'priority = 26',
                       'title = "IQ4 np1 one-row fixed-prefix target logits: pkg54 vs old path and prefill noise"',
                       'kind = "measure"', 'host_sensitive = false', 'ab = true', 'retry = "fresh"',
                       'budget_s = 6160', 'est_s = 2200', 'startup_grace_s = 90',
                       'exit_codes = { "3" = "hazard", "4" = "transient", "5" = "insufficient_data", "6" = "integrity", "7" = "timeout", "8" = "deterministic" }',
                       'signatures = [{class="hazard", regex="\\\\bXid\\\\b|fallen off the bus"}]', ''])
    text += table('run', dict(cmd=prefix + ['run', '--out', out + '/run', '--budget-s', '5800'],
                              cwd=str(TREE), ports=[], env={'GPUQ_ACTIVE': '1'}))
    text += table('canary', dict(cmd=prefix + ['canary', '--out', out + '/canary', '--budget-s', '280'],
                                 timeout_s=300, env={'GPUQ_ACTIVE': '1'}))
    text += table('preflight', dict(cmd=['bash', str(HERE / 'gpu_ab.sh'), 'preflight', '--out', str(HERE / 'evidence/queue-preflight')], timeout_s=300))
    text += table('outputs', dict(paths=[out + '/run', out + '/canary']))
    text += table('fix', dict(editable=[], protected=['frozen', 'budget', 'arms', 'noise', 'verdict']))
    text += table('pins', dict(files=sorted(set(own))))
    (HERE / 's10-kld-iq4.gpuq.toml').write_text(text.rstrip() + '\n')


def history_manifest():
    source = HERE.parent / 'identity/s10-identity-90k.toml'
    x = tomllib.loads(source.read_text())
    x.pop('diagnostic')
    x |= dict(id='s10-history-iq4', title='IQ4 np1 90k request-order dependence: s9p, pkg54 on/off, graph reuse ablations',
              priority=25, out='/mnt/scratch/work/s10-identity2/s10-history-iq4')
    old = copy.deepcopy(x['arms']['stock'])
    variants = {'s9p': ('stack9-p55', {}), 'stock': ('stack10', {}), 'off': ('stack10', {'GGML_CUDA_Q8_DECODE': '0'}),
                'stock_noreuse': ('stack10', {'LLAMA_GRAPH_REUSE_DISABLE': '1'}),
                'off_noreuse': ('stack10', {'GGML_CUDA_Q8_DECODE': '0', 'LLAMA_GRAPH_REUSE_DISABLE': '1'})}
    x['arms'] = {}
    first = []
    for variant, (name, env) in variants.items():
        for order in ('short', 'long'):
            key = variant + '_' + order
            arm = copy.deepcopy(old)
            arm |= dict(root=str(WT / 'frozen' / name), repo='', commit=kld_item.COMMITS[name], label=key,
                        canary_log=old['canary_log'][:2] if key == 'stock_long' else [])
            arm['env'] |= env
            x['arms'][key] = arm
            first.append(key)
    x['plan'] = first + list(reversed(first))
    x['depth']['wall_s'] = 300
    x['canary'] = dict(plan=['stock_long'], depths=[8192], proofs=[])
    x['budget'] |= dict(canary_s=300, margin=1.05, cleanup_s=60, est_s=4200)
    x['server']['knobs'] = list(dict.fromkeys(x['server']['knobs'] + ['GGML_CUDA_DISABLE_GRAPHS']))
    own = common_pins() + [str(source), str(HERE / 's10-history-iq4.toml')]
    x['extra_pins'] += own
    x['extra_pins'] += [str(p) for p in (HERE.parent / 'identity/results-90k').glob('full-a[23]/0-stock/depth-90112.json')]
    path = HERE / 's10-history-iq4.toml'
    path.write_text(experiment_text(x).rstrip() + '\n')
    parsed = mkitem.lib.read_experiment(path)
    mkitem.harness_snapshot = lambda _: SNAPSHOT
    text = mkitem.render(parsed)
    text = mkitem.wrap_stages(text, HERE / 'history_ab.sh')
    # The existing immutable harness snapshot holds code; the new experiment is pinned separately.
    text = text.replace(str(SNAPSHOT / 'experiment.toml'), str(path))
    lines = []
    for line in text.splitlines():
        if line.startswith('cmd = ') and ('--execute' in line or '--canary' in line):
            argv = json.loads(line.removeprefix('cmd = '))
            argv = ['bash', str(CCX), *argv]
            line = 'cmd = ' + json.dumps(argv)
        lines.append(line)
    text = '\n'.join(lines) + '\n'
    text = text.replace('kind = "measure"', 'host_sensitive = false\nkind = "measure"', 1)
    assert tomllib.loads(text)['budget_s'] <= 7200
    (HERE / 's10-history-iq4.gpuq.toml').write_text(text)


if __name__ == '__main__':
    kld_manifest()
    history_manifest()
