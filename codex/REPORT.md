# graph-reuse-safe

Retained C/D graph results and meta plans pass seeded CPU token/id/q/logit byte identity against the supplied base. Binding and epoch mutation controls fail as required. GPU validation remains parent-only.

[REPORT-graph-reuse.md](REPORT-graph-reuse.md) contains the design contract, CPU numbers, source/test evidence, model-fixture qualification, CUDA-probe compliance exception, upstream fit and risks. [gpu_ab.sh](gpu_ab.sh) queues the frozen trace-off S5 A/B; [run_reuse_cpu.sh](run_reuse_cpu.sh) reproduces CPU evidence.

Topic commits are in `codex/local.git`; `codex/graph-reuse.bundle` is the portable handoff. The prior round-trace report remains in the base commit and the read-only parent worktree.
