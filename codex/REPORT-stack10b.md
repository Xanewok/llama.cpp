# stack10b: K4 np2 crash root cause, fix, window (2026-10-09)

Tree: `~/repos/qwen38-eval-worktrees/wt/stack10b`, branch `topic/stack10b`, base `198cec743`, K9 dropped (DROPMAP recipe).

## Root cause (M on Mac ASAN, G for the exact server graph counts)

`ggml/src/ggml-backend-meta.cpp`, `ggml_backend_meta_graph_compute_impl` (stack10 line numbers):
- 2640: the subgraph arena is reallocated when `max_nnodes_raised || n_subgraphs > max_subgraphs`.
- 2653: `ctx.reset(ggml_init(...))` frees **every** cgraph in the arena.
- 2656: only `i < n_subgraphs` main graphs are recreated. `cgraphs[n_subgraphs..max_subgraphs)` keep dangling pointers.
- 2673/2677: a later graph has more subgraphs, but no more than `max_subgraphs`, so it does not reallocate. It reads `cgraph_main` from the freed arena, and `ggml_hash_set_reset(&cgraph_ij->visited_hash_set)` memsets with a garbage pointer and size. In the server GPF these were `rdi=0x3f59dd653f8703db` and `rdx=0x17e0a6bbb7e8586c`. `ggml_hash_set_reset` is a tail call, so its frame is missing from the gdb backtrace.

Why it needs K4 plus np2:
1. The RS-engaged decode graphs (n_t ≤ 32) have fewer nodes but more subgraphs than the gathered 8k prefill.
2. The warmup decodes set `max_subgraphs`.
3. The 8k prefill raises `max_nnodes` and recreates only its fewer subgraphs.
4. The first joint np2 replay reuses the spare plan and reads a freed subgraph.

So the bug is a graph ordering issue, not n_seqs as such. np1 after a long prefill is plausibly affected too (G, untested). The bug has been upstream since #22041 (`bcdcc1044`). Current master is not checked (partial clone, no lazy fetch).

## Fix commits

| Commit | Kind | Change |
|---|---|---|
| `c5087a10f` | general | recreate all `max_subgraphs` main graphs after the reset; adds `tests/test-alloc.cpp` `test_meta_subgraph_capacity(compute)` and `test_meta_subgraph_capacity(plan)` |
| `c3d17d5ef` | local | `GGML_META_SUBGRAPH_MUTANT=1` restores the pre-fix bound in the same binary; negative control only |

The test runs Meta over {CPU, CPU} with AXIS_0-split mul_mat outputs (PARTIAL). The graph sequence is: 3 subgraphs and 4000 nodes, then 1 subgraph and 8000 nodes, then 3 subgraphs and 4000 nodes. Each case runs in a forked child.

`02db31b9b` (codex) fixes the error masking in `codex/k11a/drive_np2.py` `run_window`. `capture_error` now lists the per-slot stream errors first and then appends `capture: <err>`.

## Test before/after (M, Mac M4, `stack10b-mac-mac-a2` at `47cff1bbd`)

- **ASAN `test-alloc`, fixed build:** both cases pass.
- **ASAN `test-alloc`, mutant:** fails with `ERROR: AddressSanitizer: SEGV on unknown address 0x000153463fb4` in `ggml_backend_meta_graph_compute_impl`.
  - An earlier run faulted in `_platform_memset` ← `ggml_hash_set_reset` ← `compute_impl`, the server's exact signature.
  - The fault site varies with what the freed arena holds, so the gate requires an ASAN error plus the compute_impl frame.
- **Plain CPU `test-alloc` (stage 54):** both cases pass.
- **Linux CUDA build in the window (`g_mfix`):** the fixed build must pass. The mutant rc is recorded only, because without ASAN the fault is not guaranteed.
- **ASAN np2 driver under Meta(CPU,CPU):** `fixed-on`, `mutant-on` and `mutant-off` all give rc 0 with no ASAN error. This driver's order does not reach the freed subgraph, which explains why the K4 window missed the bug. This is recorded, not gated. The GPU must-fail control is the server smoke's mutant arm (below).

## Mac CPU gates (M)

Full payload `PASS 47cff1bbd 2067 s, 52 stages`:
- k4: ops, mutants, model gate (86 s), np2 driver (127 s).
- mfix 54–57: unit 1 s, ASAN build 60 s, ASAN unit 2 s, ASAN np2 130 s.

All stages ran within ≤ 4 GiB, ≤ 10 min and load ≤ 8.

The evidence is in `codex/stack10/mac-evidence/` (commit `51b7c9039`). Raw output: `work/build-window-1007/stack10b-mac-mac-a2`.

## DROPMAP (`codex/DROPMAP.md`, `codex/stack10-drop/dropmap.tsv`)

| Component | Commits |
|---|---|
| p55 | `b28f11784 311452e04 96217fbd5 cc0bfd1b4` |
| k7 | `20ac6ec57 a4a3a8c63` |
| k4 | `386282191 6ac5f5897 6565c43db` |
| mfix | `c5087a10f c3d17d5ef` (keep whenever k4 stays) |
| k6a | `30afc9758 b4efb51a0` |
| k6b | `1376b053b 31e27ee33` |
| k2 | `fac1254d2 d045e5d49` |
| k10 | `4bf1c2ea3 9b2ea143f d1022e454` |
| k8 | `015a0cbc7 0ab5b19e7 3f0814173 c5baada6d 9ccfb56e0 7491039b9` |
| k11a | `50af425d0 a434dc77b ecb1f2524 330ab2cb8 5f1825dfa` |
| k11b | `d5b2af7c9 a8eed261a` |

k9 is removed.

## Window `codex/stack10/bw-stack10b.{toml,sh}`

- **Settings:** filler, priority 19, `host_sensitive = false`, 130 pins, estimate 30000 s, budget 60000 s.
- **Stages:** 40, 41, **43 np2-smoke (fatal)**, 42, 50, **54 meta-subgraphs**, 52, 53, 55–59, 61, 62, 64, 65.
  - The K9 gates are removed.
  - The K11a np2 sync-count gate is kept.
  - The freeze target `frozen/stack10` is unchanged; the A/Bs render as before.
  - The stock work dir is `s10b-stock-a<N>`.
- **`np2_smoke.sh` and `np2_smoke.py`:** the probe's server argv and env (IQ4_XS, `-np 2`, `-sm tensor`, ctx 204800, run under gdb).
  - The client: warm both slots, 8k prefill of both, then joint replays.
  - `fixed` arm: 3 joint replays; it must survive with full tokens on both slots.
  - `mutant` arm: `GGML_META_SUBGRAPH_MUTANT=1`, 1 replay; it must die, otherwise exit 5 (void control).
- **Checks:**
  - `env -i ... bw-stack10b.sh --preflight` printed `bw-stack10: PASS preflight` (the log prefix is kept).
  - `check_pins.py` printed `pins: 130 required files pinned, exist, none written by the window PASS`.
  - shlint gave rc 0 on all changed `.sh` files.
- **`gpuq submit --check codex/stack10/bw-stack10b.toml`:** `codex/stack10/bw-stack10b.toml: bw-stack10b: valid`.

## Uncertainty

- **GPU fix (untested):** that the fix cures the server crash on GPU is not yet measured. Stage 43 decides it, and its mutant arm must reproduce the death. The probe saw 4/4 deaths with K4 on.
- **Graph counts (G):** the exact server node and subgraph counts are not measured.
- **Review:** there was no Codex review of `c5087a10f`.
