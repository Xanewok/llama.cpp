"""Read-only phase attribution from the supplied stack6 nsys artifacts."""
import bisect
import collections
import json
import os
from pathlib import Path
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
ROOT = Path(__file__).resolve().parent.parent
SP = Path("/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad")
sys.path[:0] = ["/home/xanewok/repos/attrib-pack", str(SP / "analysis/cpu-first")]
import attrib_pack as ap
import host_sampling as hs
import np2_seg

output = {}
for np, shape in ((1, r"VCD{5}"), (2, "VCCCCCC")):
    path = f"/mnt/scratch/work/attrib-stack6{'-duo' if np == 2 else ''}-iq4/full-a1/decode-8k.sqlite"
    trace = ap.Trace(path, need_memcpy=True, need_runtime=True)
    rounds, _, _ = np2_seg.rounds_for(trace, 14, lambda value: re.fullmatch(shape, value) is not None)
    steps, windows = [], []
    phases = ["V", "C", "D1", "D2", "D3", "D4", "D5"]
    for rnd in rounds:
        wins = ap.phase_windows(trace, trace.devs[0], rnd, "llama")
        windows += [(start, end, phases[i]) for i, (_, start, end) in enumerate(wins)]
        steps += [(start, end) for start, end, kind in ap.sweep(trace, trace.devs[0], wins[0][1], wins[-1][2])[0]
                  if kind in hs.STEP_CLASSES]
    windows.sort()
    steps.sort()
    ws, ss = [w[0] for w in windows], [s[0] for s in steps]
    lo, hi = windows[0][0], windows[-1][1]
    tids = hs.decode_threads(trace, lo, hi)
    samples, chains, _, _, _ = hs.load_samples(trace.path, list(tids), lo, hi)
    spacing = hs.sample_spacing_ms({tid: [stamp for _, stamp, t in samples if t == tid] for tid in tids})
    calls = hs.api_calls(trace, set(tids), lo, hi)
    counts = collections.Counter()
    for eid, stamp, tid in samples:
        i = bisect.bisect_right(ss, stamp) - 1
        j = bisect.bisect_right(ws, stamp) - 1
        if i < 0 or j < 0 or stamp >= steps[i][1] or stamp >= windows[j][1]:
            continue
        context = hs.api_context(calls, tid, stamp)
        frames = [(hs.frame_label(s, m, u)[0], m) for s, m, u in chains.get(eid, []) if m != hs.BROKEN]
        llama_frames = [f for f, module in frames if hs.is_llama(module)]
        name = llama_frames[0] if llama_frames else (frames[0][0] if frames else "unknown")
        counts[(windows[j][2], context, name)] += 1
    copies = collections.Counter()
    for start, _, device, size, kind, *_ in trace.memcpy:
        if kind != 1:
            continue
        j = bisect.bisect_right(ws, start) - 1
        if j >= 0 and start < windows[j][1]:
            copies[(windows[j][2], device, size)] += 1
    output[f"np{np}"] = dict(artifact=path, rounds_M=len(rounds), sample_spacing_ms_M=spacing,
        host=[dict(phase=p, context=c, function=f, samples_M=n, sampled_ms_per_round_M=n*spacing/len(rounds))
              for (p, c, f), n in counts.most_common()],
        h2d=[dict(phase=p, device=d, bytes_M=b, activities_per_round_M=n/len(rounds)) for (p, d, b), n in sorted(copies.items())])
    print(f"np{np}: M rounds={len(rounds)}, sample spacing ms={spacing:.3f}")
    for (phase, context, function), count in counts.most_common():
        if phase == "C" and context == "host":
            print(f"  C host M {count*spacing/len(rounds):.3f} ms/round ({count} samples): {function}")
(ROOT / "codex/host-results/phase-attrib.json").write_text(json.dumps(output, indent=2) + "\n")
