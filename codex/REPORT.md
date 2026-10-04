# Mask upload

## Conclusion

Use an exact device-built visible prefix: existing `FILL(0)` plus `CONCAT` with a small host-built suffix. No operator, CUDA kernel, public knob, sampling change, or default CTest entry was added. The fast path is limited to the meta backend, offloaded F16 attention, one stream, at most 16 rows, and a certified prefix of at least 32768 cells. At 8192 cells it stays off; at 90112 cells with an append-only text cache, mirrored mask traffic falls from 3,964,928 to 11,264 bytes per round (D). Predicted saving is 0 ms at 8k and approximately 0.44-0.77 ms per round at 90k (D under the assumptions below), not measured GPU speed.

Base: `ff5611656cc790cf22db6281d03e226262589e6f`, branch `topic/mask-upload`. Implementation commit: `c211ca41d` (`kv-cache : build visible mask prefixes on the meta backend`). The following commit carries this report, the CPU evidence, and the parent A/B harness.

## Actual upload path

The task's async/pageable description does not match the normal path in this base:

- `src/llama-graph.cpp:29` builds the F16 graph input. `ggml/src/ggml-backend.cpp:978` places input leaves on the CPU backend. `src/llama-context.cpp:413` selects the first model device's host buffer type for that backend.
- For tensor split, the first model device is the meta device (`src/llama.cpp:176`). `ggml/src/ggml-backend-meta.cpp:387` returns the shared simple-device host type when all component devices agree. Both CUDA devices use the same `CUDA_Host` type. `ggml/src/ggml-cuda/ggml-cuda.cu:1352` uses `cudaMallocHost`; allocation failure or `GGML_CUDA_NO_PINNED` falls back to the ordinary CPU buffer at line 1362. Thus the source is normally pinned by construction; the actual recorded run's fallback state is unknown without a runtime buffer/allocator trace. Meta's false `caps.host_buffer` does not prevent this direct host-buffer lookup.
- `ggml/src/ggml-backend.cpp:1721` collects input transfers and synchronizes the destination before overwrite, then calls synchronous `ggml_backend_tensor_set_batch` at line 1739. It does not call the meta async setter at line 2031 in this path.
- `ggml/src/ggml-backend-meta.cpp:1801` fans MIRRORED batches out to each simple GPU. `ggml/src/ggml-cuda/ggml-cuda.cu:802` copies each source into that GPU's private host staging allocation. It issues `cudaMemcpyBatchAsync` on CUDA 13, or ordered individual async copies for overlapping destinations, then `cudaStreamSynchronize` at line 863. Async DMA is used inside a synchronous API.

For a contiguous F16 mask with `n_rows` total rows, each GPU receives `B = 2*n_kv*n_rows` bytes (D); there are two explicit host staging copies and two H2D payload copies (D), excluding any driver work under pinned-allocation fallback. A round of one target verify with six rows and five one-row draft decodes has `22*n_kv` bytes per GPU and `44*n_kv` mirrored bytes (D). The 16 full-attention layers reuse the mask; they do not multiply these uploads.

At 8192: one-row decode 16,384 bytes/GPU, verify 98,304 bytes/GPU, round 360,448 mirrored bytes (D). At 90112: 180,224, 1,081,344, and 3,964,928 bytes respectively (D).

`f46f8bbdb` is present and implements the synchronous batching above. Requested `53a4d8aff` is absent from both object stores; no fetch was attempted. The offline log fallback found and was read: `710b61582` (owned pinned staging slots), `6fb301b31` (synchronous host producers), and `735c2342c` (gating). Those changes still copy the complete mask separately into each GPU's owned pinned slot. They change upload lifetime and synchronization to events on a dedicated stream, not payload size or duplicate host staging. This is evidence about those available commits, not a claim about the unavailable SHA.

## Options and predictions

All savings below are D, conditional on assumptions, and exclude unrelated inputs and fixed synchronization cost. G bandwidth assumptions: effective pinned H2D bandwidth: 10-14 GB/s on PCIe Gen4 x8 and 5-7 GB/s on x4. Pageable direct-copy bandwidth: 5-9 and 3-5 GB/s. Nominal pinned case: 12 and 6 GB/s. Host staging memcpy: 20-40 GB/s. These are assumptions, not measurements of this machine. GPU memory bandwidth: 200-400 GB/s; added kernel/graph scheduling cost for the round: 0.04-0.10 ms (assumption). No P2P is available per the task's hardware facts.

| Rank at 90k | Option | Saving at 90k, ms/round (D) | Exactness / cost |
|---|---|---:|---|
| 1 by transfer ceiling | (d) Incremental device masks | About 0.44-0.83 | Could remove nearly all payload and reduce device fill traffic. Requires persistent storage and mutation/allocation generations; much larger lifecycle surface. |
| 2, chosen | (c) Device build, existing op composition | 0.44-0.77 | Rebuild a certified zero prefix and retain the exact host suffix. Stateless device contents; simple invalidation certificate. An O(rows) visible-length/future-cell custom kernel could be cheaper, but requires new ggml/backend operator support. |
| 3, possible broader upload follow-up | (a) Pinned original source | 0 as coded | Source and staging are normally already pinned, and current hooks still memcpy it twice. Teaching the transfer API to bypass staging for owned pinned sources could save 0.10-0.20, but requires a source-lifetime contract. |
| 4 | (e) Skip all-minus-infinity padded tail | At most about 0.002 before kernel costs | Ordinary KV padding is fewer than 256 cells (`src/llama-kv-cache.cpp:1250`). Must build the tail on device each time or certify it in persistent storage; otherwise stale finite values leak attention. A long retained dead suffix is a different, non-steady case. |
| 5 | (b) Upload once, peer/D2D replicate | Negative; about -0.17 nominal | Without P2P the second GPU still needs H2D and the first needs D2H via host memory. This adds a PCIe leg. A cached CUDA peer-copy path cannot remove the hardware limitation. |

With pageable direct DMA instead of this base's staged path, the same payload could take about 0.62-1.06 ms to upload at 90k (D). That is not the current baseline and must not be added to the staged-copy saving.

For (d), a sufficient fast-path invalidation rule would require unchanged device allocation epoch, tensor layout/shape/type, query row mapping and causal mode, plus a certified append-only dense single-sequence transition. Removal, membership edits, cell reuse/defrag, position/M-RoPE edits, state restore, context reset, graph allocation reuse, or changed query rows must invalidate the persistent image. Even an append changes previously future-masked cells; both the new cells and those visibility transitions must be updated. Current scheduler arenas can alias graph masks, so caching device bytes without dedicated ownership is unsafe.

Chosen path has no device-content cache. `src/llama-kv-cache.cpp:1541` certifies all occupied cells form a dense prefix and each queried sequence owns every cell in it. It reduces the prefix to the earliest future or equal-position M-RoPE-hidden cell across all rows and rounds down to KV padding. `src/llama-graph.cpp:84` rechecks the certificate before graph reuse: reuse fails if the previously generated prefix is no longer valid. An extended valid prefix need not rebuild the graph; the old suffix remains exact. The suffix setter also asserts its prefix certificate (`src/llama-kv-cache.cpp:1821`). SWA and ALiBi retain the full host path.

The original host leaf remains in `self_kq_mask`; `self_kq_mask_cnv` holds the device reconstruction. Scheduler splitting replaces graph source edges with device copies, so the setter must never recover its host pointer through a `CONCAT` edge. Both `FILL` and `CONCAT` are explicitly assigned to the meta backend after support checks (`src/llama-graph.cpp:58`); the unused shape leaf stays there, avoiding an accidental full-prefix upload.

At 90k the removed per-GPU payload is 1,976,832 bytes (D). Saving formula: `removed_bytes*(1/BW_x8 + 1/BW_x4) + 2*removed_bytes/BW_memcpy + host_fill_saved - added_device_cost`. The first term is 0.424-0.593 ms, the second 0.099-0.198 ms, host fill saves about 0.028 ms from the CPU measurements, and the estimated device cost is 0.05-0.11 ms (D from the assumptions). Nominal estimate is about 0.58 ms/round (D). At 8k, a forced compact path would remove only 0.044 ms of nominal H2D and can lose to extra kernels; the shipped gate leaves it unchanged.

## CPU evidence

`codex/check-mask-upload.sh` is standalone and is not registered with CTest. All runs used empty `CUDA_VISIBLE_DEVICES`, `nice -n 19`, at most eight threads, and no network. It builds with `cmake -B build-cpu -DGGML_CUDA=OFF` plus static libraries and OpenMP disabled, then reuses `tests/test-llama-archs.cpp`'s deterministic fixture through `codex/host-depth.cpp`.

- Exact F16 masks on `Meta(CPU,CPU)` at n_kv 8192 and 90112, one and six rows, n_pos=4, with same-position M-RoPE exclusions, tail removal, future-position edits, and sequence membership changes: PASS. The 8k composition is tested directly even though the serving threshold disables it.
- Full model graphs on a tiny Qwen35 target and MTP model with two CPU meta shards, verify/draft shape changes and cache mutations: PASS. A separate archive with graph compression forced off produced byte-identical masks and logits (`build-cpu/meta-*` versus `build-cpu/full-meta-*`). The test edits only copied source/archives under `build-cpu/mask-controls`.
- Original ff5611656 CPU artifacts versus patch: 43,646,976 mask bytes and 6,656 logit bytes match (M); SHA256 records in `codex/mask-upload-identity.json`.
- Wrong suffix injection fails with `compact mask differs from host mask`; stale host mask injection fails with `sequence mutation retained stale mask`; removing `positions.size() != n_used` from the prefix certificate in a test archive fails with `sequence mutation did not invalidate prefix`. Restoring the production archive passes. These are negative controls, not an unfixed baseline correctness bug.
- `/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`: CPU target and MTP logits finite at metadata depth 8192, n_pos=4, six-row and one-row decodes (M). Historical KV is zero and populated by metadata only; this is API/graph smoke coverage, not a real long-prompt inference benchmark. See `codex/mask-upload-real-model.txt`.

Median host setter times, microseconds (M, 101 samples with ten inner calls, same run; compact includes certificate checking):

| n_kv | Target 6 rows, full / compact | Draft 1 row, full / compact |
|---:|---:|---:|
| 8192 | 1.958 / 1.101 | 0.320 / 0.167 |
| 90112 | 17.491 / 1.894 | 2.900 / 0.397 |

Target verify plus five draft setters: 31.991 -> 3.879 us at 90k (D from M medians). CPU setter savings are not GPU measurements. The tiny Meta run emitted allocator size-estimate warnings: actual compute arena grew above the initial reservation; allocation and numerical comparisons passed.

## Parent-only GPU A/B

`codex/gpu_ab.sh` and `gpu_ab.py` retain host-step's frozen-root layout and request shapes: renamed `bin/host-depth-server`, `SOURCE`, `bin/SHA256SUMS`, root `SHA256SUMS`; 8192/90112 repeated-token prompts, ignore EOS, 512 output tokens, seed 1729, probabilistic MTP k=5, target/draft backend sampling, tensor split, q8_0 target KV and f16 draft KV. Up SOURCE is enforced as ff5611656. Full order is up/patch/patch/up. Every request is written before evaluating boot gates.

The harness checks frozen file hashes and loaded library paths, selected GPU memory, RAM availability, reusable port binding, boot identity, word-bounded kernel hazards since cutoff, exact child PID termination, wall budget, prefill/cache/length/full-K-interior metrics, native token IDs and acceptance counters. Unknown partial timing rows are skipped. Budget failures are classified as budget, not socket failures.

CPU-only `codex/test_gpu.py` passes recorded `llama-sse/long-tool-s1-1.json`, `nvidia-smi-query.csv` and `xid-lines.txt` from the queue fixtures through all available parsers, including the missing-token identity rejection. The real fixture has 24 usable intervals (M) and correctly fails the full 40-interval size gate. It has no native token IDs, so positive native-token parsing and model size gates remain parent canary checks. Frozen positive/negative controls, wrong up SOURCE, parent-only guard, budget guard, and both canary plans pass. See `codex/mask-upload-harness-test.txt`.

Queue one upstream canary boot with `--canary --canary-arm up`, then one patch canary boot with `--canary --identity-reference UP_CANARY_OUT/0-up`. Each item has one boot, the same request ordering as the full run, a maximum 600 s budget, and a 400 s stage estimate (G). The patch canary requires the upstream records and applies the same cross-arm identity gate as the full run. No same-boot repeat is inserted: it changes the MTP hidden state before the next depth. Only after both pass should the parent queue the full interleaved run, default 3600 s budget and 1920 s stage estimate (G). All reference records must be included in the queue manifest's `pins.files`, together with every file emitted in the dry plan's pins.

Example argument sequence for the parent (not executed here): `codex/gpu_ab.sh --up UPROOT --patch PATCHROOT --model MODEL --model-sha SHA --out OUT --canary --canary-arm up --execute-parent`; repeat with a new OUT, `--canary-arm patch --identity-reference FIRST_OUT/0-up`; then remove canary flags for a fresh full OUT. The queue supplies `GPUQ_ACTIVE=1` and visible devices. Without `--execute-parent`, only the pinned plan is printed. Freeze both builds before queuing; this agent did not start a server or run a GPU binary.

CUDA compile-only validation also passed: `cmake -B build-cuda -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120`, tools/tests/examples off, followed by `nice -n 19 cmake --build build-cuda --target llama -j8`. No CUDA executable was run. The build log remains at `codex/mask-upload-cuda-build.log`.

## Offline upstream check and remaining risks

Local `origin/master` is `836d57176dc699a726c55418e4f96b8ca628e1bf`; this is an offline snapshot, not a claim about today's remote. `git log origin/master -- ggml/src/ggml-backend-meta.cpp src/llama-graph.cpp src/llama-kv-cache.cpp` is saved in `codex/mask-upload-upstream-log.txt`. Relevant available changes include `631109b34` (multi-buffer allocation) and `5503b04b0` (FILL for inactive AllReduce shards). The snapshot still constructs a full host mask and has no certified-prefix upload elimination; no equivalent implementation was found in those paths. Existing F16 FILL/CONCAT CUDA support is at `ggml/src/ggml-cuda/fill.cu:33` and `ggml/src/ggml-cuda/ggml-cuda.cu:5474`.

GPU timing, CUDA capture/reuse, exact Swift token/acceptance identity, and real canary length gates remain unmeasured. The bandwidth/kernel assumptions and the prefix-size threshold need the queued A/B. Additional temporary device storage is approximately two prefix images per active graph, at most 2.16 MB per GPU for a six-row 90k mask (D, allocator reuse can reduce it); runtime reservation changes can rebuild graphs. Fragmented/multisequence, large prefill, SWA, ALiBi, unsupported backend ops and small contexts retain the host path. No upstream package tree, submission, or network lookup was created.
