# Meta plan-arena stale subgraphs (2026-10-07)

Labels: M measured, D derived, G guess. From two Sonnet code/disassembly reviews; nothing built yet.

## Defect

`ggml_backend_meta_graph_compute_impl` (`ggml/src/ggml-backend-meta.cpp`, spike ~2634, stack6 same block, upstream `7049ff0cb` 2250/2265):

- The arena is rebuilt on `max_nnodes_raised || graph_nodes_raised || ar_nodes_raised || n_subgraphs > max_subgraphs`.
- `ctx.reset(ggml_init(...))` frees the old arena, and the rebuild sizes the new one for `max_subgraphs`.
- It creates `cgraph_main` only for `i < n_subgraphs` of the current graph.
- Entries `[n_subgraphs, max_subgraphs)` keep pointers into the freed arena.
- A later graph with more subgraphs (≤ `max_subgraphs`) and no capacity raise skips the rebuild and writes through them (`n_nodes = 0`, `ggml_hash_set_reset`, `nodes[n_nodes++]`).

## Evidence of the 32k segfault (M)

- `graph-capture-depth-j5q` patch boot, kernel log: `segfault at 0 ... in libggml-base.so.0.25.3[58d25...]`.
- 0x58d25 = `mov [rcx+rax*8], r15`, i.e. `cgraph_ij->nodes[cgraph_ij->n_nodes++] = node_ij`, on the first node of a subgraph whose `nodes` read 0 through a non-null `cgraph_ij`.
- Graph kinds in node mode (`meta_ar_node` log lines), nodes, subgraphs without → with node mode, AR sites:

| owner | graph | nodes | subgraphs | sites |
|---|---|---|---|---|
| target | warm rows=2 | 4777 | 130 → 2 | 128 |
| target | prefill rows 512 | 4756 | 129 → 129 | 0 |
| target | verify/decode rows 4/6 | 4756 | 129 → 1 | 128 |
| MTP draft | warm rows 1/2 | 70 | 4 → 2 | 2 |
| MTP draft | catch-up rows 512 | 49 | 3 → 3 | 0 |
| MTP draft | rows 4/6 | 49 | 3 → 1 | 2 |

- Sequence (D): a plan first sees a prefill (`max_graph_nodes` 4756). The verify graph raises it (4756 + 128 AR nodes) and rebuilds with 1–2 subgraphs. The next 129-subgraph prefill (the 32k request) uses stale entries.
- The 8k canary is one request (prefill, then decode), so it cannot reach this order.
- There is no core dump (`coredumpctl`: none for pid 1637506).

## Reach

- **Spike (node mode):** reached, as above.
- **stack6 without node mode (D):** only `max_nnodes_raised` leaves holes. It needs a graph with more nodes but fewer subgraphs than `max_subgraphs`, followed by one with more subgraphs. The observed shapes have nodes and subgraphs positively correlated, so stack6 serving is probably not hit. It stays latent for other shapes (np2 splits, other models; G).
- **Upstream `origin/master` 7049ff0cb (D):**
  - It has the same condition and the same `i < n_subgraphs` bound, with one owner arena for all graph kinds (no `graph_plan_create`).
  - It also creates graphs with `cgraph->n_nodes` capacity instead of `max_nnodes`. Stack6 fixes that in fc1bbaf12, a cherry-pick of b22fe8e8b.
  - Neither issue is demonstrated upstream.

## Fix

- 346505678 creates every `max_subgraphs` entry on rebuild. The arena is already budgeted for it. `max_subgraphs ≤ bcj.cgraphs.size()` holds because each subgraph has ≥ 1 node.
- c3ee05e4a asserts `max_subgraphs <= max_nnodes`.
- Alternatives are worse:
  - nulling the tail entries crashes later;
  - rebuilding on every `n_subgraphs` change reallocates on each graph-kind switch;
  - lazy creation needs generation bookkeeping.
- Test `tests/test-meta-plan-arena.cpp` (e54054971, 84acdcc2e, bec62a178):
  - It runs Meta(CPU,CPU) in split-plan mode with the sequence A (7 subgraphs, 618 nodes) → B (2 subgraphs, 1203 nodes, rebuild) → A (stale entries 2..6).
  - The forked child sets `M_PERTURB` and `M_MMAP_THRESHOLD` 64 KiB, so A's arena is munmapped and the stale write faults.
  - It covers the shared defect, not the CUDA-only node plan.
- Build window `codex/bw-graph-capture-fix.sh`:
  - CPU test in both modes;
  - negative control with the fix reverted, which must die on a signal;
  - CUDA freeze;
  - rerun of the depth item (checked, not submitted).
