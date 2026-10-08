"""K11a: cudaStreamSynchronize / cudaEventSynchronize per np2 round on the decode thread, per arm (off, on, debug).
usage: count.py OUT ARM...   (OUT/<arm>/decode-8k.sqlite from the cuda pass; OUT/<arm>/server.log for the debug check)
Round segmentation and the thread choice are the X2 ones (attrib-pack llama_blocks, shape VCCCCCC, GPU0 blocks).
Gates (baseline: stack8 X2 default cuda pass, 211.9 StreamSync/round, 6.00 long):
  off:   StreamSync/round in [150, 260] (the control must show the redundant train)
  on:    StreamSync/round <= 30, long (>= 50 us) within 0.5 of the off arm and >= 5 (required syncs kept)
  debug: 'debug check engaged' logged, no 'skipped sync waited'
Exit: 0 pass, 5 fewer than 40 rounds in an arm, 8 gate failed."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/xanewok/repos/attrib-pack')
sys.path.insert(0, '/home/xanewok/repos/attrib-pack/runs/hqgap8-analysis')
import attrib_pack as ap  # noqa: E402
import an  # noqa: E402

SHAPE = 'VCCCCCC'
MIN_ROUNDS = 40


def main_tid(tr):
    c = collections.Counter(tid for s, e, nm, cid, tid in tr.runtime if nm in ap.LAUNCH)
    return c.most_common(1)[0][0]


def count_arm(path):
    tr = ap.Trace(path, need_memcpy=True, need_runtime=True)
    rounds, shapes, nall, nafter, med = an.rounds_for(tr, SHAPE)
    n = len(rounds)
    res = {'rounds': n, 'complete': nall, 'after_skip': nafter, 'median_period_ms': med / 1e6}
    if n < MIN_ROUNDS:
        return res
    main = main_tid(tr)
    d0 = tr.devs[0]
    calls = sorted((s, e, nm) for s, e, nm, cid, tid in tr.runtime if tid == main)
    import bisect
    starts = [c[0] for c in calls]
    cnt = collections.Counter()
    for r in rounds:
        seq = [b for _, b in r['phases']] + [r['next']]
        t_lo = min(k[0] for k in seq[0]['dev'][d0])
        t_hi = min(k[0] for k in seq[-1]['dev'][d0])
        for s, e, nm in calls[bisect.bisect_left(starts, t_lo):bisect.bisect_left(starts, t_hi)]:
            if nm == 'cudaStreamSynchronize':
                cnt['stream_tiny' if e - s < 5000 else 'stream_long' if e - s >= 50000 else 'stream_mid'] += 1
            elif nm == 'cudaEventSynchronize':
                cnt['event'] += 1
    for k in ('stream_tiny', 'stream_mid', 'stream_long', 'event'):
        res[k + '_per_round'] = cnt[k] / n
    res['stream_total_per_round'] = (cnt['stream_tiny'] + cnt['stream_mid'] + cnt['stream_long']) / n
    return res


def main():
    out = Path(sys.argv[1])
    arms = sys.argv[2:]
    res = {}
    for a in arms:
        log = (out / a / 'server.log').read_text(errors='replace')
        res[a] = {'debug_engaged': log.count('sync_elide: debug check engaged'), 'debug_missed': log.count('skipped sync waited')}
        res[a].update(count_arm(str(out / a / 'decode-8k.sqlite')))
    fails = []
    short = [a for a in arms if res[a]['rounds'] < MIN_ROUNDS]
    if short:
        (out / 'sync-count.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        print('FAIL insufficient rounds in', short)
        sys.exit(5)
    off, on, dbg = (res.get(a) for a in ('off', 'on', 'debug'))
    if not 150 <= off['stream_total_per_round'] <= 260:
        fails.append(f"off arm StreamSync/round {off['stream_total_per_round']:.1f} outside [150, 260]")
    if on['stream_total_per_round'] > 30:
        fails.append(f"on arm StreamSync/round {on['stream_total_per_round']:.1f} > 30")
    if abs(on['stream_long_per_round'] - off['stream_long_per_round']) > 0.5 or on['stream_long_per_round'] < 5:
        fails.append(f"required long syncs changed: off {off['stream_long_per_round']:.2f} on {on['stream_long_per_round']:.2f}")
    if dbg['debug_engaged'] < 1 or dbg['debug_missed'] != 0:
        fails.append(f"debug arm: engaged {dbg['debug_engaged']} missed {dbg['debug_missed']}")
    res['gate'] = {'pass': not fails, 'fails': fails}
    (out / 'sync-count.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print('K11a sync-count', 'PASS' if not fails else 'FAIL ' + '; '.join(fails))
    sys.exit(0 if not fails else 8)


if __name__ == '__main__':
    main()
