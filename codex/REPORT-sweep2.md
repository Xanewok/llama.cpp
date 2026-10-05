# MMVQ sweep 2

Ready for the parent's GPU queue. Source commit: 56b59124a. Production dispatch and sampling are unchanged. No GPU command or CUDA executable was run.

Added w1r4, w1r8, w2r8, w4r4, w2r6 and columns 5/7; all seven old variants remain (`tests/bench-mmvq-shapes.cpp:39`, `:84`, `:99`). The kernel permits six rows, but loads tail rows before bounding stores (`ggml/src/ggml-cuda/mmvq.cu:771`, `:828`). Each copy now has zero padding to a multiple of 24 rows (`tests/bench-mmvq-shapes.cpp:137`); timing denominators and working-set gates retain logical bytes. Maximum padding is 0.313% (D).

Compiler resource reports (M), across all types and columns:

| Variant | Registers/thread | Maximum shared bytes |
|---|---:|---:|
| w1r4 | 55-189 | 0 |
| w1r8 | 88-255 | 0 |
| w2r8 | 88-255 | 8192 |
| w4r4 | 56-193 | 12288 |
| w2r6 | 72-237 | 6144 |

Retained specializations: 175, zero stack/spill bytes (M). Dropped w1r16/w2r16: 11/10 spilling cells (M); at n=6, IQ4_XS spills 376/356 bytes and Q6_K w1r16 spills 16 bytes (M). See `sweep2/checks/resources{,-attempted}.json`. The accumulator is `float tmp[ncols][rows]` (`mmvq.cu:735`): 64 floats at the largest retained cell (D), versus 128 for sixteen rows (D). Actual occupancy is unmeasured; high registers may restrict resident warps.

Order: n=6 across every type, then 5/7/8. It ranks new variants per type at n=6, then admits complete 1/2/4 sweeps only when measured boot durations fit the remaining budget. Optional sweeps include base/w2r4 controls. Tables report GB/s, best variant and gains over both controls. Missing mandatory data or incomplete optional data produces no tables. CUDA/reference disagreement is exit 6 (integrity); the perturbed-output negative control must fail. Exact-PID child cleanup and kernel hazard checks are included.

Checks: `bash codex/sweep2/build.sh` built only bench-mmvq-shapes, -j6, nice 19, local CCCL v3.4.3, arch 120a, NCCL OFF, Release. RAM stayed above the 10 GiB floor. CPU preflight passed in 1.06 s (M):

CUDA_VISIBLE_DEVICES="" SWEEP2_PREFLIGHT_ROOT="$PWD/codex/sweep2" OUT="$PWD/codex/sweep2/checks/preflight" nice -n 19 bash codex/sweep2/run.sh --preflight

 Parser selftests and recorded-data tests passed on 1085 primary + 62 repeat rows (M), old env/negative logs and the queue's hazard fixture. Remapped recorded rows test new schemas only. CPU child normal/error/timeout cleanup passed; timeout took 1.002 s (M). Manifest schema, all pins and budgets passed; evidence is in `sweep2/checks/`.

After copying the entire package, including ignored frozen binaries, to its final location:

gpuq submit /home/xanewok/repos/qwen38-eval/results/2026-10-05-mmvq-sweep2/mmvq-sweep2.toml

Canary cap 240 s + run cap 900 s = 1140 s (D), within the 1200 s item budget. Estimated total 800 s (G). GPU correctness, occupancy and speed remain open; canary gates the full run. Padding may affect old-cell comparisons. Primary warmup matches the old run; secondary warmup is shorter.
