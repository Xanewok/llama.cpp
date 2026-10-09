"""fable_bcd_summary.py OUT -> verdict.txt, summary.tsv, served.tsv in OUT (rc 0 verdicts computed, 5 control void, 6 integrity, 8 input error).

Inputs in OUT, all printed by tests/test-fable-bcd.cpp (ROW dev proc lever config arm round trim_us median_us total_us host_us):
  run-CUDA1.tsv, run-CUDA0.tsv   levers B, C, D on each GPU (proc "main")
  run-nopdl-CUDA1.tsv            lever C, stock arms only, GGML_CUDA_PDL=0 (diagnostic: does PDL overlap exist in the chain at all)
The metric is trim_us (10 % trimmed mean of the per-rep graph interval, µs). CUDA1 (headless) is the verdict device, CUDA0 the replicate.

Rules, fixed before the run (brief 2026-10-09; report codex/REPORT-fable-bcd.md):
  Per config, control A and its identical repeat A2 (B: cold_a/cold_b, C: stock_a/stock_b, D: K6_a/K6_b) are measured every round.
  S (A/A spread) = max - min over all 2R control samples. base_r = mean(A_r, A2_r). Effect of arm X in round r: eff_r = base_r - X_r
  (positive = X faster). X "fires" when min_r eff_r > S: every repeat exceeds the spread. Otherwise the effect is "none" (inside the A/A).
  B   exists   pf10 fires on >= 50 % of the 8 configs. VOID (rc 5) if the positive control `warm` (no flush) fires on < 75 % of configs
               (the baseline is not cold) or the decoy `wrong10` (prefetch of another matrix) fires on > 25 % (the gain is not L2 retention
               of the target). Saved µs per MB = median eff / MB prefetched.
  C   class exists when pf1 fires on >= 50 % of the class's configs; "exists (deeper only)" when pf1 does not but pf2, pf4 or pf16 does;
               else none. Classes: MMVQ n=6 (iq4_xs, q6_K x 3 chains), MMVQ n=1 (draft), MMQ n=12 (K9-fix PDL substrate). Per link = eff / links.
               VOID if S > 2 % of base on > 50 % of configs. Diagnostic (not void): stock with PDL off vs on.
  D   per (mode, shape): K1 and K3 vs K6. Per-sequence saving = eff / n_seqs / 1000 ms per round (the graph is the 48 GDN layers of one round).
               exposed >= 0.6 ms/seq and fires; partial 0.2..0.6 and fires; else none (Fable thresholds). Primary mode is "mixed"
               (each scan followed by two weight GEMMs so DRAM is busy), "iso" is the upper bound. Dose check: eff(K1) > eff(K3) > 0.
               VOID if S > 2 % of base on > 50 % of the D configs.
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RATIO_VOID = 0.02
ROUNDS_MIN = 3


def load(path):
    rows, cfg, route, meta = defaultdict(dict), {}, {}, []
    done = False
    for line in Path(path).read_text().splitlines():
        f = line.split('\t')
        if f[0] == 'ROW' and len(f) == 11:
            _, dev, proc, lever, config, arm, rnd, trim, med, tot, host = f
            rows[(lever, config)].setdefault(arm, {})[int(rnd)] = (float(trim), float(med), float(tot), float(host))
        elif f[0] == 'CFG':
            cfg[(f[3], f[4])] = dict(kv.split('=', 1) for kv in f[5:])
        elif f[0] == 'ROUTE':
            route[(f[3], int(f[4]))] = int(f[5])
        elif f[0] == 'META':
            meta.append(f)
        elif f[0] == 'DONE':
            done = True
    return rows, cfg, route, meta, done


class Cfg:
    def __init__(self, lever, name, arms, ctrl):
        self.lever, self.name, self.arms, self.ctrl = lever, name, arms, ctrl
        a, a2 = ctrl
        rounds = sorted(set(arms[a]) & set(arms[a2]))
        if len(rounds) < ROUNDS_MIN:
            raise ValueError(f'{lever} {name}: only {len(rounds)} rounds of the control')
        self.rounds = rounds
        ctl = [arms[a][r][0] for r in rounds] + [arms[a2][r][0] for r in rounds]
        self.spread = max(ctl) - min(ctl)
        self.base_r = {r: (arms[a][r][0] + arms[a2][r][0]) / 2 for r in rounds}
        self.base = statistics.mean(self.base_r.values())

    def eff(self, arm):
        """-> (median effect, min effect over rounds, fires)"""
        v = [self.base_r[r] - self.arms[arm][r][0] for r in self.rounds]
        return statistics.median(v), min(v), min(v) > self.spread

    def host_bound(self):
        worst = max(max(self.arms[a][r][3] / self.arms[a][r][0] for r in self.rounds) for a in self.arms)
        return worst


def configs(rows, lever, ctrl):
    out = {}
    for (lv, name), arms in rows.items():
        if lv == lever:
            out[name] = Cfg(lever, name, arms, ctrl)
    return out


def verdict_device(dev, path, lines, tsv, served, nopdl_path=None):
    rows, cfg, route, meta, done = load(path)
    void = []
    if not done:
        raise SystemExit(f'{path}: no DONE line (run did not finish)')
    rounds = int(meta[0][7].split('=')[1])
    res = {}
    # ---- B
    B = configs(rows, 'B', ('cold_a', 'cold_b'))
    if len(B) != 8:
        raise SystemExit(f'{dev}: B has {len(B)} configs, expected 8')
    warm_fire = sum(c.eff('warm')[2] for c in B.values())
    wrong_fire = sum(c.eff('wrong10')[2] for c in B.values())
    lines.append(f'{dev} B controls: warm fires {warm_fire}/8 (need >= 6), wrong10 fires {wrong_fire}/8 (need <= 2)')
    if warm_fire < 6:
        void.append(f'{dev} B: warm control fires on {warm_fire}/8 configs: baseline is not cold')
    if wrong_fire > 2:
        void.append(f'{dev} B: decoy prefetch fires on {wrong_fire}/8 configs: gain is not L2 retention of the target')
    per_mb = defaultdict(list)
    for name, c in sorted(B.items()):
        n = int(name.rsplit('_n', 1)[1])
        info = cfg[('B', name)]
        for arm in ('warm', 'pf5', 'pf10', 'pf19', 'wrong10', 'bulk10', 'ldcg10'):
            med, mn, fires = c.eff(arm)
            mb = {'pf5': 'pf5', 'pf10': 'pf10', 'pf19': 'pf19', 'wrong10': 'pf10', 'bulk10': 'pf10', 'ldcg10': 'pf10'}.get(arm)
            mbytes = int(info[mb]) / 2**20 if mb else 0.0
            tsv.append(f'{dev}\tB\t{name}\t{arm}\t{c.base:.2f}\t{med:+.3f}\t{mn:+.3f}\t{c.spread:.3f}\t{int(fires)}\t{mbytes:.2f}\t'
                       f'{(med / mbytes if mbytes else 0):+.3f}')
            if arm in ('pf5', 'pf10', 'pf19') and fires:
                per_mb[arm].append(med / mbytes)
    b_fire = sum(c.eff('pf10')[2] for c in B.values())
    res['B'] = 'exists' if b_fire >= 4 else 'none'
    lines.append(f"{dev} B verdict: {res['B']} (pf10 fires on {b_fire}/8 configs); us saved per MB when firing: "
                 + ', '.join(f'{k} median {statistics.median(v):+.2f} (n={len(v)})' for k, v in sorted(per_mb.items())))
    for mode in ('bulk10', 'ldcg10'):
        lines.append(f'{dev} B {mode} fires on {sum(c.eff(mode)[2] for c in B.values())}/8')
    # ---- C
    C = configs(rows, 'C', ('stock_a', 'stock_b'))
    if len(C) != 12:
        raise SystemExit(f'{dev}: C has {len(C)} configs, expected 12')
    noisy = sum(c.spread > RATIO_VOID * c.base for c in C.values())
    lines.append(f'{dev} C A/A: S > {RATIO_VOID:.0%} of base on {noisy}/12 configs')
    if noisy > 6:
        void.append(f'{dev} C: A/A spread above {RATIO_VOID:.0%} on {noisy}/12 configs')
    classes = {'MMVQ n=6': [], 'MMVQ n=1': [], 'MMQ n=12': []}
    for name, c in sorted(C.items()):
        n = int(name.rsplit('_n', 1)[1])
        links = int(cfg[('C', name)]['links'])
        classes['MMVQ n=6' if n == 6 else 'MMVQ n=1' if n == 1 else 'MMQ n=12'].append(c)
        for arm in ('stock_b', 'pf1', 'pf2', 'pf4', 'pf16'):
            med, mn, fires = c.eff(arm)
            tsv.append(f'{dev}\tC\t{name}\t{arm}\t{c.base:.2f}\t{med:+.3f}\t{mn:+.3f}\t{c.spread:.3f}\t{int(fires)}\t{links}\t{med / links:+.4f}')
    for t, n in (('iq4_xs', 6), ('q6_K', 6), ('iq4_xs', 1), ('iq4_xs', 12)):
        want = 0 if n == 12 else 1
        got = route.get((t, n))
        if got != want:
            raise SystemExit(f'{dev}: routing {t} n={n}: mmvq={got}, expected {want}')
    cres = {}
    for cls, cs in classes.items():
        f1 = sum(c.eff('pf1')[2] for c in cs)
        fd = max(sum(c.eff(a)[2] for c in cs) for a in ('pf2', 'pf4', 'pf16'))
        half = (len(cs) + 1) // 2
        cres[cls] = 'exists' if f1 >= half else 'exists (deeper only)' if fd >= half else 'none'
        per_link = [c.eff('pf1')[0] / int(cfg[('C', c.name)]['links']) for c in cs]
        lines.append(f'{dev} C {cls}: {cres[cls]} (pf1 fires {f1}/{len(cs)}, deeper best {fd}/{len(cs)}); '
                     f'pf1 us per link median {statistics.median(per_link):+.3f}, range {min(per_link):+.3f}..{max(per_link):+.3f}')
    res['C'] = cres
    if nopdl_path and Path(nopdl_path).exists():
        nrows, _, _, _, ndone = load(nopdl_path)
        if ndone:
            N = configs(nrows, 'C', ('stock_a', 'stock_b'))
            fire, tot = 0, 0
            for name, c in C.items():
                if name in N:
                    tot += 1
                    off = statistics.mean(N[name].base_r.values())
                    fire += (off - c.base) > max(c.spread, N[name].spread)
            lines.append(f'{dev} C diagnostic: PDL on is faster than PDL off by more than the spread on {fire}/{tot} configs '
                         f'(0 => no ramp overlap to hide a prefetch under)')
    # ---- D
    D = configs(rows, 'D', ('K6_a', 'K6_b'))
    if len(D) != 4:
        raise SystemExit(f'{dev}: D has {len(D)} configs, expected 4')
    noisy = sum(c.spread > RATIO_VOID * c.base for c in D.values())
    lines.append(f'{dev} D A/A: S > {RATIO_VOID:.0%} of base on {noisy}/4 configs')
    if noisy > 2:
        void.append(f'{dev} D: A/A spread above {RATIO_VOID:.0%} on {noisy}/4 configs')
    dres = {}
    for name, c in sorted(D.items()):
        n_seqs = int(name.rsplit('_s', 1)[1])
        mode = name.split('_')[1]
        m1, mn1, f1 = c.eff('K1')
        m3, mn3, f3 = c.eff('K3')
        per_seq = m1 / n_seqs / 1000.0
        verdict = 'none' if not f1 or per_seq < 0.2 else 'partial' if per_seq < 0.6 else 'exposed'
        dose = 'ok' if m1 > m3 > 0 else 'absent'
        dres[(mode, n_seqs)] = (verdict, per_seq, m1 / 1000.0)
        tsv.append(f'{dev}\tD\t{name}\tK1\t{c.base:.2f}\t{m1:+.3f}\t{mn1:+.3f}\t{c.spread:.3f}\t{int(f1)}\t{n_seqs}\t{per_seq:+.4f}')
        tsv.append(f'{dev}\tD\t{name}\tK3\t{c.base:.2f}\t{m3:+.3f}\t{mn3:+.3f}\t{c.spread:.3f}\t{int(f3)}\t{n_seqs}\t{m3 / n_seqs / 1000.0:+.4f}')
        lines.append(f'{dev} D {name}: {verdict}; K6->K1 {m1:+.1f} us per round-graph ({per_seq:+.3f} ms/seq/round), fires={int(f1)}, '
                     f'K6->K3 {m3:+.1f}, dose-response {dose}, base {c.base:.0f} us, S {c.spread:.1f}')
    res['D'] = dres
    for c in list(B.values()) + list(C.values()) + list(D.values()):
        hb = c.host_bound()
        if hb > 0.8:
            lines.append(f'{dev} WARN {c.lever} {c.name}: host enqueue per rep up to {hb:.0%} of the GPU interval (graph launch may be host-bound)')
    # ---- served impact (D = derived; call counts G)
    served.append(f'# {dev}: ms per verify round and GPU, derived from measured us x call counts (G: counts from the Fable report and restrict_summary.py)')
    b_np1 = statistics.median(B[n].eff('pf10')[0] for n in B if n.endswith('_n6'))
    b_np2 = statistics.median(B[n].eff('pf19')[0] for n in B if n.endswith('_n12'))
    served.append(f"{dev}\tB\tnp1_k5\t128 AR calls x pf10 median {b_np1:+.2f} us\t{b_np1 * 128 / 1000:+.3f}\tverdict={res['B']}")
    served.append(f"{dev}\tB\tnp2_k5\t128 AR calls x pf19 median {b_np2:+.2f} us\t{b_np2 * 128 / 1000:+.3f}\tverdict={res['B']}")

    def link(cfgname):
        c = C[cfgname]
        return c.eff('pf1')[0] / int(cfg[('C', cfgname)]['links'])

    for t in ('iq4_xs', 'q6_K'):
        n6 = [link(f'{t}_{ch}_n6') for ch in ('updown', 'sq5120', 'small')]
        np1 = (192 * n6[0] + 48 * n6[1] + 112 * n6[2] + 16 * statistics.mean(n6) + 1 * statistics.mean(n6)) / 1000
        served.append(f"{dev}\tC\tnp1_k5 {t} n=6\t369 target GEMMs (192 up/down chain, 48 sq5120, 112 small chain, 17 other at the mean)\t{np1:+.3f}\t{cres['MMVQ n=6']}")
    n1 = statistics.mean(link(f'iq4_xs_{ch}_n1') for ch in ('updown', 'sq5120', 'small'))
    served.append(f"{dev}\tC\tnp1_k5 draft n=1\t25 draft GEMMs x mean pf1 {n1:+.3f} us\t{25 * n1 / 1000:+.3f}\t{cres['MMVQ n=1']}")
    n12 = [link(f'iq4_xs_{ch}_n12') for ch in ('updown', 'sq5120', 'small')]
    np2 = (192 * n12[0] + 48 * n12[1] + 112 * n12[2] + 17 * statistics.mean(n12)) / 1000 + 50 * n1 / 1000
    served.append(f"{dev}\tC\tnp2_k5\t369 target GEMMs at 12 columns (MMQ, K9-fix substrate) + 50 draft GEMMs at n=1\t{np2:+.3f}\t{cres['MMQ n=12']}")
    for (mode, ns), (v, per_seq, tot) in sorted(dres.items()):
        label = 'np1_k5' if ns == 1 else 'np2_k5'
        served.append(f'{dev}\tD\t{label} {mode}\t48 GDN layers, K6->K1, before the replay cost (D 0.37 ms/seq, Fable)\t{tot:+.3f}\t{v}')
    return res, void


def main():
    out = Path(sys.argv[1])
    lines, tsv, served = [], ['dev\tlever\tconfig\tarm\tbase_us\tmedian_eff_us\tmin_eff_us\taa_spread_us\tfires\tunit\tper_unit'], []
    served.append('dev\tlever\tprofile\tbasis\tms_per_round\tverdict')
    void = []
    results = {}
    for dev in ('CUDA1', 'CUDA0'):
        p = out / f'run-{dev}.tsv'
        if not p.exists():
            raise SystemExit(f'missing {p}')
        res, v = verdict_device(dev, p, lines, tsv, served, out / 'run-nopdl-CUDA1.tsv' if dev == 'CUDA1' else None)
        results[dev] = res
        void += v
    lines.append('--- replicate agreement (CUDA1 verdict device, CUDA0 replicate)')
    a, b = results['CUDA1'], results['CUDA0']
    lines.append(f"B: {a['B']} / {b['B']}")
    for cls in a['C']:
        lines.append(f"C {cls}: {a['C'][cls]} / {b['C'][cls]}")
    for k in sorted(a['D']):
        lines.append(f"D {k[0]} seqs={k[1]}: {a['D'][k][0]} / {b['D'][k][0]}")
    if void:
        lines.append('VOID: ' + '; '.join(void))
    (out / 'verdict.txt').write_text('\n'.join(lines) + '\n')
    (out / 'summary.tsv').write_text('\n'.join(tsv) + '\n')
    (out / 'served.tsv').write_text('\n'.join(served) + '\n')
    print('\n'.join(lines), file=sys.stderr)
    sys.exit(5 if void else 0)


if __name__ == '__main__':
    main()
