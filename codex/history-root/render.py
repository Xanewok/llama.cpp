"""Render the item and pin its local imports, frozen bytes, and recorded parser artifacts."""
import json
from pathlib import Path
import sys
import tomllib

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
WT = TREE.parents[1]
BASE = TREE / 'codex/stack10/identity2'
sys.path.insert(0, str(BASE))
from render_kld_v2 import value


def render():
    old = tomllib.loads((BASE / 's10-kld-iq4-v2.gpuq.toml').read_text())
    item = {k: v for k, v in old.items() if k not in ('run', 'canary', 'preflight', 'outputs', 'pins', 'fix')}
    item.update(id='history-root', priority=23,
                title='Teacher-forced request history: first differing tensor, clears, draft and graph ablations',
                budget_s=7000, est_s=4500)
    output = '/mnt/scratch/work/history-root/a{attempt}'
    wrapper = str(WT / 'work/aa-load/ccx01.sh')
    for stage, budget in [('run', 6600), ('canary', 280)]:
        item[stage] = {'cmd': ['bash', wrapper, 'bash', str(HERE / 'gpu_ab.sh'), stage,
                              '--out', output + '/' + stage, '--budget-s', str(budget)],
                       'env': {'GPUQ_ACTIVE': '1'}}
    item['run'].update(cwd=str(TREE), ports=[])
    item['canary']['timeout_s'] = 300
    item['preflight'] = {'cmd': ['bash', str(HERE / 'gpu_ab.sh'), 'preflight', '--out',
                                str(HERE / 'evidence/queue-preflight')], 'timeout_s': 300}
    item['outputs'] = {'paths': [output + '/run', output + '/canary']}
    item['fix'] = {'editable': [], 'protected': ['frozen', 'budget', 'arms', 'reading']}
    files = {p.replace('/wt/s10-identity/', '/wt/history-root/') for p in old['pins']['files']
             if '/wt/s10-identity/codex/stack10/identity2/' not in p}
    files |= {str(BASE / name) for name in ('kld_item_v2.py', 'render_kld_v2.py')}
    files |= {str(TREE / 'codex/gpu_ab.sh')}
    files |= {str(p) for p in HERE.iterdir() if p.suffix in ('.py', '.cpp', '.sh') or p.name == 'CMakeLists.txt'}
    files |= {str(TREE / p) for p in ('common/common.h', 'common/arg.h', 'include/llama.h', 'src/llama-ext.h',
                                     'ggml/include/ggml.h', 'ggml/include/ggml-backend.h', 'ggml/include/ggml-cpp.h',
                                     'ggml/src/ggml-backend-impl.h', 'ggml/src/ggml-impl.h')}
    files |= {str(WT / 'bin/cpugate')}
    files |= {'/home/xanewok/models/hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/swift_draft_vocab_ids.json'}
    item['pins'] = {'files': sorted(files)}
    text = ''.join(f'{k} = {value(v)}\n' for k, v in item.items() if not isinstance(v, dict))
    for table, data in item.items():
        if isinstance(data, dict):
            text += f'\n[{table}]\n' + ''.join(f'{json.dumps(k)} = {value(v)}\n' for k, v in data.items())
    return text


if __name__ == '__main__':
    (HERE / 'history-root.gpuq.toml').write_text(render())
