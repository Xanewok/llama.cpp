"""Reuse the frozen depth harness; keep request order as an explicit arm dimension."""
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'gpu'))
sys.path.insert(0, str(HERE.parent / 'identity'))
import gpu_s10
import compare

depth, lib = gpu_s10.depth, gpu_s10.lib
ORDERS = {'short': [8192, 90112], 'long': [8192, 32768, 90112]}


def analyze(experiment, out):
    x = lib.read_experiment(experiment)
    records = {}
    for i, arm in enumerate(x['plan']):
        record = compare.read_record(Path(out) / f'{i}-{arm}/depth-90112.json')
        counts = compare.counters(record)
        if counts['cache_n'] != 0 or counts['prompt_n'] != 90112:
            raise lib.Fail('90k request reused prompt cache', 'integrity')
        if arm in records and not compare.matches(compare.compare(records[arm], record)):
            raise lib.Fail('same-order repeat identity failed', 'integrity')
        records[arm] = record
    result = {}
    for base in ('s9p', 'stock', 'off', 'stock_noreuse', 'off_noreuse'):
        row = compare.compare(records[base + '_short'], records[base + '_long'])
        result[base] = dict(order_dependent=not compare.matches(row), **row)
    (Path(out) / 'history-diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def preflight(experiment):
    x = lib.read_experiment(experiment)
    a = compare.read_record(HERE.parent / 'identity/results-90k/full-a2/0-stock/depth-90112.json')
    b = compare.read_record(HERE.parent / 'identity/results-90k/full-a3/0-stock/depth-90112.json')
    assert a['request'] == b['request'] and not compare.matches(compare.compare(a, b))
    for record in (a, b):
        result, ident = depth.gate(x['depth'], record, depth.first_prefill(90112), k=5)
        assert result['timings']['cache_n'] == 0 and len(ident[0]) == 512
    mutated = json.loads(json.dumps(a))
    mutated['chunks'][0]['tokens'][0] += 1
    assert not compare.matches(compare.compare(a, mutated))
    assert all(x['arms'][name]['env']['LLAMA_MTP_FOLD_C'] == '1' for name in x['plan'])
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        for i, arm in enumerate(x['plan']):
            dest = Path(tmp) / f'{i}-{arm}'
            dest.mkdir()
            (dest / 'depth-90112.json').write_text(json.dumps(a if arm.endswith('_short') else b))
        assert all(v['order_dependent'] for v in analyze(experiment, tmp).values())
        last = Path(tmp) / f"19-{x['plan'][19]}/depth-90112.json"
        last.write_text(json.dumps(mutated))
        try:
            analyze(experiment, tmp)
        except lib.Fail as error:
            assert error.rc == 6
        else:
            raise AssertionError('repeat identity negative control missed')
    print('history parser: real 90k short/long, repeated record, token mutation, cache and sufficiency gates passed')


def main():
    argv = sys.argv[1:]
    if '--preflight' in argv:
        preflight(argv[1])
        return 0
    gpu_s10.install(argv)
    original_boot = depth.boot
    def boot(x, arm, port, dest, canary, stub=False):
        x['depth']['depths'] = ORDERS[arm['name'].rsplit('_', 1)[1]]
        return original_boot(x, arm, port, dest, canary, stub)
    depth.boot = boot
    original_finish = depth.finish
    depth.finish = lambda x, results: original_finish(x, {90112: results[90112]})
    rc = gpu_s10.base.harness_main()
    if rc == 0 and '--execute' in argv:
        analyze(argv[1], argv[argv.index('--out') + 1])
    return rc


if __name__ == '__main__':
    try:
        sys.exit(main())
    except lib.Fail as error:
        print(str(error), file=sys.stderr)
        sys.exit(error.rc)
