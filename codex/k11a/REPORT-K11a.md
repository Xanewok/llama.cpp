# K11a getter sync elision (tree wt/s10-k11a, branch topic/s10-k11a, base 0cfc0723e)

Labels: M measured, D derived, G guess.

## Commits (general first, `local:` and test/window commits interleaved by time; only src/llama-context.{cpp,h} change)
- d72c180b5 llama : skip getter synchronize() when no device work is pending (the general change)
- 63c4f13f7, 6e60fa241, 3f302cb46, 393cbef5c `local:` env switches: LLAMA_SYNC_ELIDE=0 (off), LLAMA_SYNC_ELIDE_DEBUG=N (real sync after each skipped one, log if it waited > 20 us, N>1 sets the threshold), LLAMA_SYNC_ELIDE_MUTANT=1|2|3 (negative controls)
- 950a616b7, 430c0bf04, e5e500c06, 1af42b31a Mac payload `codex/bw-k11a-mac.sh` + HOST_DEFER arm in `codex/bw-np2-driver.cpp`
- c0329951a window: `codex/bw-k11a.{sh,toml}`, `codex/k11a/{sync_count.sh,count.py,drive_np2.py}`
- a3140b9c4 Mac evidence `codex/k11a/mac-evidence/` (summary.json source hash is checked by the window preflight)
- Incidental source changes: none beyond the above (no compile fixes needed).

## Design (general commit)
`llama_context::backend_pending` (initially true). Set before `ggml_backend_sched_graph_compute_async` in `graph_compute` and in the one helper `llama_context::tensor_get_async` that wraps every `ggml_backend_tensor_get_async` (decode and output paths, `logits_deferred_fetch`, `extract_layer_inputs`; `copy_tensor_async_rows` gets a `bool &` and sets it). Cleared only in `synchronize()` after the real `ggml_backend_sched_synchronize`. `synchronize(bool force=false)` returns at once when `!force && !backend_pending && n_queued_tokens == 0 && sched n_copies == 1`. The n_copies guard keeps the `next_copy = 0` side effect of the real sync (pipeline parallel always syncs). Lifecycle syncs (destructor, `sched_reserve`, `output_reserve` realloc) use `force`. Getters (`llama_get_*`, `llama_synchronize`) are unchanged and keep calling `synchronize()`; public semantics "outputs visible on return" hold.
Exactness: no kernel, copy, order or sampling change; only host-side stream-sync calls are skipped on a stream that this context knows to be drained. The stats block (`n_queued_tokens`) is a no-op when nothing is queued, so it is unchanged. Assumptions (D): one thread uses a context (as before for `n_queued_tokens`); the context's own backends/streams are the only ones its getters wait on (backends are per context; the MTP fold writes into the dft cache from the target's streams, and a dft getter never waited for those before either). Enqueue sites grep: `graph_compute_async` and `tensor_get_async` occur only in `src/llama-context.cpp` (loader's upload backend is separate; KV/memory ops use synchronous `ggml_backend_tensor_*`).
Open risk (D): a future enqueue added without the helper is missed; the `local:` debug mode is the guard on GPU runs.

## Mac results (M, `work/build-window-1007/k11a-mac-a4`, HEAD 1af42b31a, Qwen3.5-4B Q4 CPU, 48 tokens, seed 1729)
- build 70 s; identity 436 s; mutants 434 s; rc 0.
- Token streams and draft/accept counters identical across four arms {LLAMA_SYNC_ELIDE=0, default, deferred backend + off, deferred backend + debug} in all five combos (np,Meta,foldC) = 1,0,1 / 1,1,0 / 2,0,0 / 2,1,0 / 2,1,1 (np1 and np2, plain and Meta(CPU,CPU), fold on and off).
- Elisions per context 432-1280 per run; debug check `missed=0` everywhere.
- HOST_DEFER (driver) makes the CPU backend stream-like: graph computes and device-to-host copies queue until `synchronize`, so a skipped needed sync reads stale host data. It engaged (defer busy syncs 104-7815 per run) and the stock path passes under it, so the deferred backend itself is valid.
- Negative controls (3 combos 2,0,0 / 1,0,1 / 2,1,0): mutant 3 (flag cleared after every enqueue) is caught by token mismatch in all 3 combos and by the debug check in all 3; mutants 1 and 2 (drop one of the two marks) are NOT caught by tokens and by the debug check only in some combos (mut1: 2,0,0 and 2,1,0; mut2: 2,0,0 and 1,0,1). Reason (D): on the exercised paths the other mark covers the getter (graph_compute always followed by a get_async; the sync-get_async-sync pair in `logits_deferred_fetch` is not reached with backend sampling), so these two are benign mutants, not an escaped stale read. The `logits_deferred_fetch` mark has no dedicated test (blocker: needs a run that calls `llama_get_logits` with deferred logits).
- Debug-threshold fix: the first Mac run (a2) logged one false `missed` of 23 us on an empty queue (scheduler noise, 4 contexts, load 8-100); Mac gates use N=80 vs a 100 us simulated wait; GPU window keeps 20 us.
- Not run: CUDA, np1 GPU, Metal.

## Window (not submitted)
`codex/bw-k11a.toml` (id bw-k11a, filler, priority 18, preemptible=false, host_sensitive=false, same GPU UUIDs/recipe 120a, CUDA 13.2, NCCL off, CCCL 3.4.3, DeviceTopK count 25, `ports = "dynamic"`). Stages: 40 CUDA build of llama-server (+ `round-trace` copy); 41 stamps (commit, DeviceTopK, debug string); 42 `codex/k11a/sync_count.sh`: three arms booted in turn on the same binary under `nsys launch` (off = LLAMA_SYNC_ELIDE=0, on = default, debug = LLAMA_SYNC_ELIDE_DEBUG=1), each with a proof boot (env reached the server, kv_pad, ub 512, no CPU sampler, chain/fold off) and the X2 np2 decode-8k driver, then `count.py` gates.
`count.py` was validated on the X2 default cuda trace: 205.58 tiny + 0.28 mid + 6.00 long = 211.86 StreamSync/round, 280 EventSync/round (X2: 211.9 / 6.00 / 280, M).
Gates: off arm 150-260 StreamSync/round; on arm <= 30 and long (>= 50 us) within 0.5 of the off arm and >= 5; debug arm "debug check engaged" logged and no "skipped sync waited". Preflight (Mac evidence hash, dry run of the server script) passes on this host.
`gpuq submit --check`: `bw-k11a: valid`; shlint rc 0 on `bw-k11a.sh`, `k11a/sync_count.sh`, `bw-k11a-mac.sh`.

## Expected effect (D from X2 section 5a; G realised)
206 -> ~14 StreamSync per round (198 redundant removed; 6 long required + ~7 second-rank calls stay). Critical path at the five interior D boundaries: train of 27 tiny calls spans 41.7-44.9 us, minus the required pair: -0.20 ms/round D, -0.12..-0.20 G (np2 8k stack8). Depth-flat (counts do not depend on KV), so the same at 90k; with K13 (chain) 4 of the 5 trains disappear and the saving shrinks to ~0.04.
Falsifier: on-arm StreamSync > 60/round, or interior D gap not shorter by >= 0.02 ms (use X2's `x2_tiny.py` / `c_step.py` on the new trace), or any token difference in a later A/B (baseline-vs-baseline negative control first). This window does not time the gain; the timing verdict needs an off/on pair in the stack item.
