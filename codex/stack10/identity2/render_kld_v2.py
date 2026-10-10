"""Render only the repaired KLD item; history inputs remain pinned."""
import json
from pathlib import Path
import tomllib

HERE = Path(__file__).resolve().parent


def value(v):
    if isinstance(v, dict):
        return '{ ' + ', '.join(json.dumps(k) + ' = ' + value(x) for k, x in v.items()) + ' }'
    if isinstance(v, list):
        return '[' + ', '.join(value(x) for x in v) + ']'
    return json.dumps(v)


def render():
    source = HERE / 's10-kld-iq4.gpuq.toml'
    item = tomllib.loads(source.read_text())
    item['id'] = 's10-kld-iq4-v2'
    old_out = '/mnt/scratch/work/s10-identity2/s10-kld-iq4/'
    new_out = '/mnt/scratch/work/s10-identity2/s10-kld-iq4-v2/'
    for stage in ('run', 'canary', 'preflight'):
        item[stage]['cmd'] = [arg.replace(str(HERE / 'gpu_ab.sh'), str(HERE / 'kld_v2_ab.sh'))
                             .replace(old_out, new_out)
                             .replace('evidence/queue-preflight', 'evidence/kld-v2/queue-preflight')
                             for arg in item[stage]['cmd']]
    item['outputs']['paths'] = [p.replace(old_out, new_out) for p in item['outputs']['paths']]
    own = ['kld-window-v2.cpp', 'kld-v2/CMakeLists.txt', 'kld_item_v2.py', 'kld_v2_ab.sh',
           'render_kld_v2.py', 'test_kld_v2.py', 'kld_item.py',
           'evidence/kld-v2/history-pins-before.json', source.name]
    item['pins']['files'] = sorted({p for p in item['pins']['files'] if not p.startswith(str(HERE) + '/')}
                                  | {str(HERE / p) for p in own})
    text = ''.join(f'{k} = {value(v)}\n' for k, v in item.items() if not isinstance(v, dict))
    for table, data in item.items():
        if isinstance(data, dict):
            text += f'\n[{table}]\n' + ''.join(f'{json.dumps(k)} = {value(v)}\n' for k, v in data.items())
    return text


if __name__ == '__main__':
    (HERE / 's10-kld-iq4-v2.gpuq.toml').write_text(render())
