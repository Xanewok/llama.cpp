"""Render the draft-vocab-ext GPU depth items (never submits). Usage: gen_items_vocab_ext.py CPU_RESULTS_DIR.

Arms: frozen stack6 vs frozen draft-vocab-ext with --spec-draft-vocab-ext 512; J5q TP2 np1 k=5 ub512 on the real-prompt
depth harness (wt/harness-real-prompt: the random-token fixture's acceptance is meaningless). mkitem runs from that tree.
Item 1 uses tokens64k: its 6 sessions are tasks of the draft-vocab-cov analysis corpus (in-sample). Item 2 uses tokens8k
when its 16-gram overlap with tokens64k and the analysis prompts is below 5 % (held-out candidate); otherwise it is skipped.
"""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
W = Path('/home/xanewok/repos/qwen38-eval-worktrees')
EVAL = Path('/home/xanewok/repos/qwen38-eval')
HARNESS_TREE = W / 'wt/harness-real-prompt'
FROZEN = W / 'frozen/draft-vocab-ext'
KLD = EVAL / 'recipes/_common/kld'
ANALYSIS_PROMPTS = EVAL / 'results/2026-10-06-spec-trace/full-a2'
N_EXT = 512
ENGAGED = r'draft vocab ext: seq 0: \d+ prompt OOV ids, [1-9]\d* in the draft head'
ROWS = r'draft vocabulary extension: 512 rows after \d+ reduced-head rows'


def value(obj):
    if isinstance(obj, dict):
        return '{ ' + ', '.join(f'{json.dumps(str(k))} = {value(v)}' for k, v in obj.items()) + ' }'
    if isinstance(obj, list):
        return '[' + ', '.join(value(v) for v in obj) + ']'
    return json.dumps(obj)


def write_toml(path, data):
    path.write_text('\n'.join(f'{key} = {value(obj)}' for key, obj in data.items()) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kldt(path):
    data = Path(path).read_bytes()
    assert data[:4] == b'KLDT', path
    count, off, out = struct.unpack_from('<i', data, 4)[0], 8, []
    for _ in range(count):
        n = struct.unpack_from('<i', data, off)[0]
        out.append(list(struct.unpack_from(f'<{n}i', data, off + 4)))
        off += 4 + 4 * n
    return out


def grams(tokens, n=16):
    return {tuple(tokens[i:i + n]) for i in range(0, max(0, len(tokens) - n + 1))}


def heldout_overlap():
    seen = set()
    for seq in kldt(KLD / 'tokens64k.bin'):
        seen |= grams(seq)
    for p in sorted(ANALYSIS_PROMPTS.glob('p*.tokens.json')):
        seen |= grams(json.loads(p.read_text())['tokens'])
    cand = set()
    for seq in kldt(KLD / 'tokens8k.bin'):
        cand |= grams(seq)
    return len(cand & seen) / max(1, len(cand)), sum(len(s) for s in kldt(KLD / 'tokens8k.bin'))


def local_pins():
    return [str(HERE / n) for n in ('DESIGN-vocab-ext.md', 'gen_items_vocab_ext.py', 'bw-vocab-ext.sh', 'bw-ext-compare.py')]


def render(path):
    subprocess.run([sys.executable, '-m', 'harness.mkitem', str(path), '--out', str(path.with_suffix('.gpuq.toml'))],
                   cwd=HARNESS_TREE, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))


def main():
    cpu_results = Path(sys.argv[1]).resolve()
    evidence = cpu_results / 'summary.txt'
    assert evidence.exists(), f'{evidence} missing: run the CPU stage first'
    commit = (FROZEN / 'SOURCE').read_text().splitlines()[0]
    template = tomllib.loads((HARNESS_TREE / 'harness/experiments/realprompt-depth-j5q.toml').read_text())
    s6 = template['arms']['s6']
    ext = copy.deepcopy(s6)
    ext.update(label=f'draft-vocab-ext {commit[:9]} (stack6 + extension), --spec-draft-vocab-ext {N_EXT}, --kv-pad 4096, CCCL343, staging',
               commit=commit, root=str(FROZEN), repo=str(ROOT))
    ext['flags'] = s6['flags'] + ['--spec-draft-vocab-ext', str(N_EXT)]
    ext['canary_log'] = s6['canary_log'] + [ENGAGED, ROWS]
    up = copy.deepcopy(s6)
    up['label'] = 'stack6 ac7465885, --kv-pad 4096, CCCL343, staging (draft head 25879 ids, no extension)'
    up['canary_log'] = s6['canary_log'] + [r'(?s)\A(?!.*draft vocab ext: )']
    day = datetime.date.today().isoformat()
    (HERE / 'items').mkdir(exist_ok=True)

    overlap, n8k = heldout_overlap()
    print(f'tokens8k: {n8k} tokens, 16-gram overlap with tokens64k + analysis prompts {100 * overlap:.2f} %')
    corpora = [('tokens64k', [8192, 32768, 90112], 'in-sample: the 6 sessions are tasks of the draft-vocab-cov analysis corpus')]
    if overlap < 0.05 and n8k >= 32768 + 1024:
        corpora.append(('tokens8k', [8192, 32768], f'held-out candidate: 16-gram overlap {100 * overlap:.2f} % with the analysis prompts and tokens64k'))
    else:
        print('tokens8k item skipped: not held out or too short')

    for corpus, depths, note in corpora:
        x = copy.deepcopy(template)
        name = f'draft-vocab-ext-np1-j5q-{corpus}'
        x['id'] = name
        x['title'] = (f'stack6 vs per-request draft vocab extension ({N_EXT} rows from the prompt), J5q TP2 np1 k=5 ub512, ABBA, real prompts '
                      f'({corpus}; {note}); judge tokens/round, acceptance, ms/token; identity within each arm (the ext pool '
                      f'depends on request history, but every boot loads the same filler rows and runs the same depths in order)')
        x['plan'] = ['up', 'ext', 'ext', 'up']
        x['arms'] = {'up': up, 'ext': ext}
        x['out'] = str(EVAL / f'results/{day}-{name}')
        x['extra_pins'] = local_pins() + [str(KLD / f'{corpus}.bin')]
        x['cpu_evidence'] = {'path': str(evidence), 'command': 'codex/bw-vocab-ext.sh stage 1 (test-draft-vocab-ext single/tp + mutant, bw-ext-driver matrix)'}
        d = x['depth']
        d['depths'] = depths
        # np1: "off" is refused (streams > 1 only). The arms draft differently, so tokens differ across arms; within an
        # arm the pool history is the same in every boot, so each arm must repeat its own tokens and counters
        d['identity'] = 'within-arm'
        d.pop('identity_off_reason', None)
        d['distinct_builds'] = True
        d['prompt'] = {'path': str(KLD / f'{corpus}.bin'), 'sha': sha256(KLD / f'{corpus}.bin')}
        d['wall_s'] = 300 if len(depths) == 3 else 220
        x['canary'] = {'plan': ['up', 'ext'], 'depths': [8192], 'proofs': []}
        x['budget'].update(canary_s=600, est_s=1500 if len(depths) == 3 else 1100)
        path = HERE / 'items' / f'{name}.toml'
        write_toml(path, x)
        render(path)
        print(f'rendered {path.with_suffix(".gpuq.toml")}')


if __name__ == '__main__':
    main()
