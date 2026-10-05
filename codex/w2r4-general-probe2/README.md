Run only through the parent GPU queue. `gpu_ab.sh` delegates to this kernel-suite harness; no server is started.

The full order is up/patch/patch/up on each GPU. Up is the existing frozen `98c695111` bundle in q8-1row; patch is the new general freeze. Both arms run aligned CUDA-vs-CPU correctness and the same perf cases. Only patch runs the partial-row regression cases, because the baseline contains known invalid reads on these shapes. No intended-fault stage is included.

The canary is one patch boot with all aligned, partial and perf geometries. It exercises the same parsers, shape/commit/device/completeness gates, hashes, library resolution, memory-release wait and kernel-log checks as the full item. Every process command, stdout, stderr and exact owned PID is saved before gate evaluation. Correctness failures are recorded and all stages in the boot are evaluated before failure. No comparison is written after a failed gate.

Budgets: G 80 s perf, G 10 s aligned correctness, G 15 s partial correctness per boot; D full stage estimate 780 s, within 70% of the 1800 s run budget. G canary 105 s, within 70% of its 300 s limit. Total queue budget 2400 s includes the 360 s canary stage limit. Actual timings remain canary checks. `--only correctness` and `--only perf` select stages; canary always checks all stages.

Recorded parser fixtures come from the real MMVQ baseline run and gpuq inventory/kernel-log artifacts. SQL fixtures retain their measured fields and geometry; only case-name prefixes are adapted to this harness. Negative controls reject failed correctness, invalid timing, missing/duplicate records, changed geometry, backend and commit. `selftest` and `--preflight` run without GPU binaries.

The test process has CPU affinity limited to eight CPUs and niceness 19. Kernel processes inherit queue visibility, and cleanup signals only the Popen-owned PID. GPUq provides admission, host-RAM and boot-ID guards.

Submit from the parent after review:

    gpuq submit /home/xanewok/repos/mmvq-w2r4/codex/w2r4-general-probe/item.toml

Only `gpuq submit --check` is authorized in this worktree session. GPU token identity and end-to-end latency need a separate serving A/B after this correctness/perf gate.
