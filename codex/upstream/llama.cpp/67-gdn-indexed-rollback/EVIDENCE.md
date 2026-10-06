# Getter-form evidence

Primary source: master `7049ff0cb` -> `topic/gdn-getter` source head `472791d90`. The staged patch series recreates that tree exactly (`codex/logs/apply-tree.txt`, `source-tree.txt`). Stack6 source: `ac7465885` -> `topic/gdn-getter-stack6` `743acb55f`, frozen at `codex/frozen/gdn-getter-stack6/` with the matching ggml stamp. CUDA builds pass; no GPU executable was run.

M denotes measured results. CPU-only validation through cpugate:

| Check | Result | Artifact |
|---|---|---|
| GDN op tests | M57/57 pass, including M13 indexed equivalence and M3 indexed cache-copy cases | `codex/logs/backend-cpu.log` |
| Row-0 negative control | M13/13 indexed cases fail; negative-control verdict PASS | `codex/logs/negative-row0.{log,json}` |
| Architecture fixture | M133/133 pass; explicit context 256 | `codex/logs/archs-cpu.{log,json}` |
| Qwen3.5-4B rollback/checkpoint, np1 and np2 | PASS on patch and base | `codex/logs/model-{patch,base}-np{1,2}.log` |
| Idle child after seq_cp, rollback enabled, NO_REALLOC | PASS on patch; extra-state relocation required by one-child decode | `codex/logs/model-patch-np{1,2}.log` |
| Split replay | FAIL identically on patch and base: Mmax difference 13.4578, MNMSE 0.018913, first seq 0 pos 16 | same model logs |

Stack6 `--state-movers` passes at np1/np2, with its negative controls (M8/M16 trial-sequence results), `codex/logs/stack-state-movers-np{1,2}.{log,json}`. The renamed CPU MTP server passes NO_REALLOC, np2, context 4096, positive rollback and seq_cp log gates; `codex/logs/reserve-server.{log,json}`, `codex/logs/reserve-server/`.

Parser tests on real recorded sanitizer, SSE and trace artifacts pass, including negative controls. CPU-only frozen/input preflights pass. Both final manifests pass `gpuq submit --check`: `codex/logs/{memcheck,depth}-contract.log`. No item was submitted or run on a GPU. The regular model suite stops after cache fill 0x00 because of the base split-replay failure, so it does not reach fill 0x3e.

The index input selects the same float row as GET_ROWS and leaves the GDN arithmetic and sampling unchanged. CPU comparisons check the residual against zero, not just CPU versus itself. The negative control leaves the gather branch intact and forces indexed reads to row 0.

GPU items are prepared, not submitted: `codex/memcheck/item.toml` (separate sanitizer log files, memcheck/racecheck/synccheck) and `codex/depth-item.toml` (frozen stack6 versus getter, ABBA, J5q np1 k5, cross-arm native identity and indexed engagement). Getter GPU correctness, sanitizer cleanliness and latency are unknown. Existing opcode-form evidence lives under `alt-opcode/` and is not evidence for this getter form.

Full commands, API/backend locations, CPU validation times, stack-port provenance and remaining risks: `codex/REPORT-gdn-getter.md` in the assigned worktree. The parent installs this staged replacement in canonical package 67.
