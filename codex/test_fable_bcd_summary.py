"""Self-test of fable_bcd_summary.py on synthetic runs: null data must give 'none' everywhere (negative control), injected
effects must give 'exists', a broken control must give rc 5. Run: python3 codex/test_fable_bcd_summary.py"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = 7


def gen(dev, proc, inject, warm_fires=True, decoy_fires=False, noise=0.3, nopdl=False, seed=1):
    rng = random.Random(seed)
    out = [f'META\t{dev}\t{proc}\tl2_bytes=33554432\tflush_bytes=67108864\tnsm=36\tlever=all\trounds={R}\tstock_only=0']
    for t, n in (('iq4_xs', 6), ('q6_K', 6), ('iq4_xs', 1), ('iq4_xs', 12), ('q6_K', 12)):
        out.append(f'ROUTE\t{dev}\t{proc}\t{t}\t{n}\t{0 if n == 12 else 1}')

    def rows(lever, config, base, arms, extra):
        out.append(f'CFG\t{dev}\t{proc}\t{lever}\t{config}\t' + '\t'.join(extra))
        for r in range(R):
            for arm, eff in arms.items():
                v = base - eff + rng.gauss(0, noise)
                out.append(f'ROW\t{dev}\t{proc}\t{lever}\t{config}\t{arm}\t{r}\t{v:.4f}\t{v:.4f}\t{v + 300:.4f}\t5.0')
    if not nopdl:
        for t in ('iq4_xs', 'q6_K'):
            for shape in ('8704x5120', '5120x5120'):
                for n in (6, 12):
                    e = inject.get('B', 0.0)
                    arms = {'cold_a': 0, 'cold_b': 0, 'warm': 30 if warm_fires else 0, 'pf5': e / 2, 'pf10': e, 'pf19': e * 1.5,
                            'wrong10': 20 if decoy_fires else 0, 'bulk10': e, 'ldcg10': e}
                    rows('B', f'{t}_{shape}_n{n}', 80.0, arms, ['weight_bytes=24000000', 'spin_ns=23000', 'pf5=5242880', 'pf10=10485760', 'pf19=19922944'])
    for t, n in (('iq4_xs', 6), ('q6_K', 6), ('iq4_xs', 1), ('iq4_xs', 12)):
        for ch in ('updown', 'sq5120', 'small'):
            key = 'C12' if n == 12 else 'C'
            e = inject.get(key, 0.0)
            arms = {'stock_a': 0, 'stock_b': 0}
            if not nopdl:
                arms.update({'pf1': e, 'pf2': e, 'pf4': e, 'pf16': e})
            base = 2000.0 + (60 if nopdl else 0)
            rows('C', f'{t}_{ch}_n{n}', base, arms, ['links=48', 'weights_per_pool=3', 'bytes_each=1'])
    if not nopdl:
        for mode in ('iso', 'mixed'):
            for ns in (1, 2):
                e = inject.get('D', 0.0) * ns
                rows('D', f'gdn_{mode}_t6_s{ns}', 5000.0, {'K6_a': 0, 'K6_b': 0, 'K3': e / 2, 'K1': e}, ['layers=48'])
    out.append(f'DONE\t{dev}\t{proc}')
    return '\n'.join(out) + '\n'


def run(inject, **kw):
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for i, dev in enumerate(('CUDA1', 'CUDA0')):
            (d / f'run-{dev}.tsv').write_text(gen(dev, 'main', inject, seed=i + 1, **kw))
        (d / 'run-nopdl-CUDA1.tsv').write_text(gen('CUDA1', 'nopdl', {}, nopdl=True, seed=9))
        p = subprocess.run([sys.executable, str(HERE / 'fable_bcd_summary.py'), str(d)], capture_output=True, text=True)
        return p.returncode, p.stderr


def check(name, cond):
    print(('ok   ' if cond else 'FAIL ') + name)
    if not cond:
        check.bad += 1


check.bad = 0
rc, err = run({})
check('null data: rc 0', rc == 0)
check('null data: B none', 'CUDA1 B verdict: none' in err)
check('null data: C none in every class', err.count('C MMVQ n=6: none') == 3 and err.count('C MMQ n=12: none') == 3)
check('null data: D none', 'gdn_mixed_t6_s1: none' in err and 'gdn_iso_t6_s2: none' in err)
rc, err = run({'B': 10.0, 'C': 1.5, 'C12': 1.5, 'D': 300.0})
check('injected: rc 0', rc == 0)
check('injected: B exists', 'CUDA1 B verdict: exists' in err)
check('injected: C exists', 'CUDA1 C MMVQ n=6: exists' in err and 'CUDA1 C MMQ n=12: exists' in err)
check('injected: D exposed per seq 0.3 ms is partial', 'gdn_mixed_t6_s1: partial' in err)
rc, err = run({'D': 1000.0})
check('injected: D 1.0 ms/seq exposed', 'gdn_mixed_t6_s2: exposed' in err)
rc, err = run({'B': 10.0}, warm_fires=False)
check('warm control silent: void rc 5', rc == 5 and 'baseline is not cold' in err)
rc, err = run({'B': 10.0}, decoy_fires=True)
check('decoy fires: void rc 5', rc == 5 and 'not L2 retention' in err)
rc, err = run({}, noise=60.0)
check('noisy A/A: void rc 5', rc == 5)
sys.exit(1 if check.bad else 0)
