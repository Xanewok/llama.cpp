# Indexed GDN rollback state

## Conclusion

Implemented on stack6 `ac7465885dc9f13de396ccd91b4be5eaffde3730`. The GDN kernel reads the selected snapshot row directly through the existing recurrent-copy input. Eligible graphs omit the state `GET_ROWS`; snapshot production, convolution snapshots, rollback bookkeeping and sampling are unchanged.

CPU evidence passes, including Meta(CPU,CPU), distinct per-sequence rollback depths and stale-index controls. CUDA compiles and is frozen at `38c59b36872e44a7f8ddd1a183acacb599796b2d`. No GPU execution or GPU performance measurement was performed. The checked depth item is ready for the parent to queue.

CUDA currently enables indexed reads only for a single sequence per microbatch. Multiple-sequence CUDA batches retain the existing gather and fused snapshot stores. CPU and Meta(CPU,CPU) support indexed reads for multiple sequences.

## Flow and design

`src/llama-memory-recurrent.cpp:101` allocates the live plane and rollback planes in the existing recurrent cache. `seq_rm` at `:200` records a bounded, single-use rollback distance in `rs_idx`; a second pending rollback is rejected. `s_copy` at `:1364` produces `rs_idx * mem->size + src0` and consumes `rs_idx`. Previously `build_rs` gathered these rows into a compact tensor before GDN. GDN then wrote its produced snapshots to the same physical plane layout.

`ggml/include/ggml.h:2666` adds `ggml_gated_delta_net_indexed`. Its state input is contiguous F32 `[S_v, S_v, H_v, n_state_rows]`; its row input is contiguous I32 `[n_seqs]`. The output layout and arithmetic match `ggml_gated_delta_net`. The new opcode is appended to preserve existing opcode values. A separate opcode lets backend capability checks retain the gather path on unsupported backends, including RPC (`ggml/src/ggml-rpc/ggml-rpc.cpp:2222`).

`src/models/delta-net-base.cpp:548` selects the indexed op only with snapshots enabled, recurrent offload allowed, `n_rs == n_seqs`, and backend support. Extra recurrent rows still go through `build_rs`: their relocation can overwrite a source row before direct reads finish. An unbound host buffer uses the CPU backend capability check. Other unsupported buffers/backends retain the original path.

`ggml/src/ggml-cpu/ops.cpp:11212` changes only the state-input row offset. `ggml/src/ggml-backend-meta.cpp:862` keeps the index tensor mirrored while splitting state heads. `ggml/src/ggml-cuda/gated_delta_net.cu:45` waits for the PDL dependency before reading the row input. CUDA fused output writes remain in place (`ggml/src/ggml-cuda/ggml-cuda.cu:3039`). For multiple sequences, one destination can alias another input row; fusing those writes with indexed reads is unsafe without a device-wide read barrier. The capability check at `:6037` therefore keeps those batches on the existing gather path rather than introducing extra snapshot copies.

The existing input tensor carries the selected row. Acceptance changes input data, not graph tensor dimensions, op count or output layout. Existing changes in prompt/decode token counts still select their usual graph shapes; this patch does not promise one graph across different token counts.

### Invariants and cases

- Full draft acceptance uses rollback plane zero, which already contains the latest state.
- Zero draft acceptance retains the mandatory base token. For verify width six (D, k plus one), removing five draft tokens selects rollback plane five. Partial acceptance selects the corresponding existing plane.
- Full `seq_rm`, including a clear while rollback is pending, discards that selection. The original zero-row clearing operation is preserved before the next state read.
- Prompt batches use the same kernel and final snapshot rules. When prompt length exceeds snapshot capacity, only the existing last snapshots are kept.
- Sequence reordering changes row-index values while output sequence order remains the batch order. CPU np2 covers unequal accepted counts, reversed batch order and shared-tail copies. CUDA np2 is correct through the gather fallback.
- Convolution state remains on stack6's fused `SSM_CONV` path (`src/models/delta-net-base.cpp:449`). No recurrent allocation, checkpoint format or memory-module indexing rule changes.
- The sampling chain and its RNG are untouched. Exact op comparisons plus matching logits, MTP hidden-state fingerprints and sampled tokens verify the CPU path; GPU end-to-end identity remains a required A/B gate.

## Evidence

All executed commands hid GPUs with `CUDA_VISIBLE_DEVICES=""`; git used `GIT_NO_LAZY_FETCH=1`. Compiles ran through `bin/buildgate`, nice 19, CMake `-j4`. CPU model and multicore tests ran through `bin/cpugate`, nice 19, with at most eight workers and explicit context 4096. Baseline libraries were built before source edits and are verified against `codex/base-cpu/SHA256SUMS`.

| Check | Result |
| --- | --- |
| `gdn-op-cpu` against explicit `GET_ROWS` | M: 192 byte-identical comparisons; M: 192 wrong-slot controls detected |
| CPU `test-backend-ops` executable renamed `gdn-backend-cpu` | M: 56/56 tests passed, including M: 15 indexed/cache-fusion cases |
| Real 4B model, CPU and Meta(CPU,CPU), np1 and np2 | M: all four baseline/patch transcript comparisons passed |
| Wrong-slot and stale-after-`seq_rm` model controls | M: both changed the compared output |
| CPU RPC backend compilation | Passed; indexed capability intentionally rejected |
| CUDA 13.2, SM120a, NCCL off, local CCCL343 | Compiled server and backend-op test; neither executed |
| GPU item rendering and `gpuq submit --check` | Passed; no submission |
| Read-only GPU preflight | Passed frozen hashes, model/vocabulary hashes and recorded parser fixtures |

`codex/gdn-cpu.cpp` performs scripted target verification and rollback, not the complete speculative server loop. It loads the MTP head, checks actual target hidden states, samples with the pinned seed, and compares FNV64 fingerprints of every retained recurrent layer after each phase. It covers zero/partial/full draft acceptance, pending-rollback rejection, checkpoint save/restore and clear/re-prompt. Model comparisons use fingerprints rather than raw tensor bytes, so hash collisions are a residual limitation; the standalone op comparison is byte-exact.

CPU timings below are M, milliseconds inside the fixture after model/context loading. They include prompt, verification, state reads, rollback/checkpoint/clear phases and sampling; they are single observations, not a speed benchmark. CPU np2 additionally exercises `seq_cp`; the Meta np2 path omits that optional case due to the baseline fault described below.

| Backend / sequences | stack6 wall ms (M) | patch wall ms (M) | indexed layer calls (M) |
| --- | ---: | ---: | ---: |
| CPU / np1 | 3961.073 | 4372.735 | 264 |
| CPU / np2 | 7310.255 | 6847.466 | 432 |
| Meta(CPU,CPU) / np1 | 4782.018 | 4757.782 | 264 |
| Meta(CPU,CPU) / np2 | 8899.753 | 8912.064 | 360 |

Reproduce the CPU components with `CUDA_VISIBLE_DEVICES="" GIT_NO_LAZY_FETCH=1 nice -n 19 bash codex/run-gdn-cpu.sh`. The retained baseline libraries must be present at `codex/base-cpu/bin`; their source and checksums are committed, their binaries are local artifacts. `run-backend-cpu.sh` caps the existing test's hardware-concurrency probe so reference and backend workers together stay within the thread limit.

The stock `cpugate` status command could not access the writable gpuq lock/socket in this sandbox. `codex/gate-bin/gpuq` supplies only read-only `status --json` from the authoritative queue files under a shared lock. Any read failure reports a running measure item, so the CPU gate waits. It is not used for submission validation.

### Baseline fault

Meta(CPU,CPU) np2 with the additional `seq_cp` shared-tail case crashes in stack6 before the next model graph. The backtrace reaches `ggml_backend_tensor_set_2d` from `llama_kv_cache::update`, not GDN (`codex/cpu-results/meta-np2-fault.txt`; `src/llama-kv-cache.cpp:817`). Repeated baseline runs, including graph/Meta reuse disabled, failed. Required np2 rollback, clear and checkpoint tests still pass; the optional case is enabled with `GDN_TEST_META_SEQ_CP=1`. This unrelated fault is not fixed here.

## GPU A/B handoff

- Experiment: `codex/depth-experiment.toml`; rendered item: `codex/depth-item.toml`; wrapper: `codex/gpu_ab.sh`.
- Baseline: frozen stack6 `ac7465885`; patch: `codex/frozen/gdn-gather-inplace`, source `38c59b368`. Both use `--kv-pad 4096` and `LLAMA_ARG_KV_PAD=4096`, J5q, TP2, k5 and cross-arm token identity.
- Full plan: ABBA at depths 8192, 32768 and 90112. Canary: one patch boot at depth 8192, timeout 600 seconds (D). Rendered total budget is 3450 seconds (D, inherited wall estimate plus margin and cleanup); it is not measured here.
- The standard depth harness supplies frozen-file/hash gates, exact-PID lifecycle, record-before-gate ordering, bounded Xid journal checks and steady full-k intervals. No harness source was edited.
- `GGML_CUDA_DEBUG_GDN_STATE=1` requires indexed verify-six logs on each GPU. `gdn_gather_skipped` counts indexed layer reads on each graph compute, including graph replays. Existing convolution-fusion proofs remain required. New log lines have only a synthetic positive test until the parent runs the canary; an actual unpatched recording fails this engagement gate.

`codex/test-gpu-parsers.py` passed against the real stack6 depth recording at `/home/xanewok/repos/qwen38-eval/results/2026-10-05-stack6-depth-j5q/full-a1/1-s6/depth-8192.json` and the required recorded fixtures. It validates SSE/timings/native tokens, cross-arm identity mutation, graph-counter parsing, memory CSV and Xid parsing. The committed trace excerpt and provenance are in `codex/parser-fixtures/`; metrics in `parser-results.json` belong to that old recording, not this patch.

Validated commands: `python3 -m harness.mkitem ... --out codex/depth-item.toml` from the eval repository, `bash codex/gpu_ab.sh --check`, `bash codex/gpu_ab.sh --preflight`, and `python3 codex/test-gpu-parsers.py`. Preflight uses an explicit unused port to avoid the harness's free-port socket probe in this sandbox. No canary, execute stage or CUDA test was run.

## Predicted cost and remaining risks

The supplied baseline attribution is M: 0.30 ms per round per GPU for the gather. For Swift's dimensions, D: a state plane is 72 MiB per GPU across the GDN layers; removing its gather avoids roughly 144 MiB of read/write traffic per GPU per round. D: verify-six snapshot writes remain 432 MiB per GPU per round. Those writes are not removed. D: the attributable gather budget is about 0.30 ms on the round's critical path; GPU contributions must not be added together. G: net improvement is unknown until the A/B, because input locality, graph scheduling and trace overhead can change the result.

CUDA op exactness, graph replay with changing indices, fused aliasing, np2 fallback and real engagement remain untested on a GPU. The new trace scan/events run only when round tracing is enabled and add patch-side host overhead to the A/B. Other backend fallback paths are supported by capability checks but were not runtime-tested here. The new opcode changes the op count; the RPC version patch/count assertion is updated, and RPC rejects the opcode so older peers retain the original graph.

## Local prior art and upstream split

Local `git log -S`, no network, found `c56031644` (reuse SSM graphs), `66625a59a` (reduce recurrent/hybrid splits), `3688c4f50` (Kimi-Linear recurrent input flow), and `5a460dea9` (remove redundant CUDA copies after GDN). This follows their existing input and fused-output mechanisms; it does not introduce a new memory subsystem. HyperQwen's accepted-slot approach is task-provided background, not independently verified here.

For upstream, keep the indexed GGML API, CPU/Meta capability handling, CUDA kernel/fusion guards, model builder integration and backend-op tests together. Keep the local trace instrumentation and `codex/` harness/evidence out of that patch. Stack6's snapshot/fusion prerequisites and backend runtime coverage must be established before proposing that split. No second upstream package tree, PR description, submission or remote action was created.

Source commits: `03fe64c15` GGML/API/CPU/Meta/model integration and tests; `c51652aa1` CUDA indexed reads; `f5c5bd54f` local engagement tracing; `9e999126b` CUDA multi-sequence fallback; `38c59b368` CPU fixtures/evidence and freeze tooling. The later report/item commit does not alter the frozen implementation.
