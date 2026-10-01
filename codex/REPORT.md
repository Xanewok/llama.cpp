# Task E: CPU rollback discrepancy

Verdict (a), high confidence: the reported discrepancy comes from batch-dependent CPU arithmetic. No wrong recurrent, conv, KV retention, position, or snapshot-slot defect was established. Use both `GGML_CPU_REPACK=OFF` and `GGML_LLAMAFILE=OFF` for this CPU rollback oracle. No production or sampling code was changed.

M = executed measurement or observation. D = source/arithmetic inference. G = guess; no performance guesses are made here. Configuration arguments, token IDs, hashes and source locations are literals.

## Decisive evidence

All measurements use the supplied `Qwen3.5-4B-UD-Q4_K_XL.gguf`, full target head, teacher-forced token histories. The main control uses f16 KV, FA off, native CPU instructions, OpenMP, no BLAS and no repacking. Reference and rollback contexts start from the same serialized prompt state. The reference evaluates only retained tokens and never rolls them back.

| Check | Result (M) | Evidence |
| --- | --- | --- |
| Requested prefixes `120..140`, `250..260`; configured k `1..6`; every rollback distance | 672 cases; zero pre-correction logit, conv, GDN and post-correction logit NMSE | `sweep-exact.log`, `exact-summary.json` |
| Prefix `257`, every k and distance, final byte oracle | 21 cases; byte-identical logits, conv/GDN tensors and serialized logical state including retained KV/positions | `state-bytes-strict.log` |
| k `5`, distance `5`, prefix `257`, continuation | All 200 subsequent teacher-forced logit rows byte-identical to the independent reference | `follow-strict.log` |
| Original supplied fixture, q8 KV / FA on, single and Meta CPU, k `5`, prefix `257` | Every distance passes with zero logit NMSE after disabling SGEMM | `original-q8-single-no-sgemm.log`, `original-q8-meta-no-sgemm.log` |
| Deliberately select slot `r-1` instead of `r` | Exit 1; conv NMSE 0.471167677, GDN NMSE 0.0269419436, post-correction NMSE 0.163853968 | `wrong-slot-strict-negative.log` |

The initial complete sweep checked float-value equality; the final `memcmp` oracle was subsequently run on every prefix-`257` rollback case and the continuation. It detects different signed-zero bytes even when NMSE is zero. All continuation rows are checked; only selected checkpoints are printed. The Meta CPU version of the original fixture uses its existing host sampling fallback for the tied target head.

With the read-only native build (`GGML_LLAMAFILE=ON`, repacking still off), the original failure reproduces exactly:

| k `5`, distance `5`, prefix `257` | NMSE (M) |
| --- | ---: |
| Retained-token logits before any rollback can affect them | 0.000912029576 |
| Selected conv state versus single-token reference | 0.000135852667 |
| Selected GDN state versus single-token reference | 0.0000240548043 |
| After corrected token | 0.00158219451 |
| After continuation step `50` | 0.00115052832 |
| After continuation step `100` | 0.00315685302 |
| After continuation step `200` | 0.00111854422 |
| Maximum over the continuation | 0.0635568835 |
| Rollback versus matched multi-token computation, all state comparisons and continuation steps | 0 |

Persistence or growth is not a sufficient discriminator here. The native error grows while the independently evaluated multi-token shape control remains equal to the rolled result. That control evaluates `keep+1` tokens, checks its retained row before rollback, then removes just its last token. It uses distance one rather than the tested maximum distance. It matches CPU arithmetic dispatch, not the exact full verification column count; the independent no-rollback reference under SGEMM-off supplies the strict state oracle.

M: native prefix-`257` tests have 15 equal cases and 6 differing cases: precisely the maximum rollback at every configured k. Native k-`5` maximum rollback differs at every one of the 32 requested prefix lengths; post-correction NMSE ranges from 0.000676568452 to 0.00198421302. Their matched-control NMSE is always zero. Changing prompt ubatch from `128` to `64` leaves the decisive corner's pre/post errors unchanged. Thus neither k `5` nor the prompt boundary alone is required (D).

## Design and source audit

`codex/state_split_probe.cpp:162` computes `keep=k+1-r`. It reads the selected conv/GDN planes directly (`:56`) before correction, checks retained position and `rs_idx`, and compares complete serialized logical state (`:188`) when requested. The negative control changes the actual pending slot (`:170`), not just the output oracle. The continuation uses identical tokens in each context (`:198`). `state_summary.py --sweep` rejects missing or duplicated cases; its artifact controls reject perturbed/truncated data and distinguish signed zero.

D: `ggml/src/ggml-cpu/llamafile/sgemm.cpp:3820` rejects right-hand matrix column counts below `2`. `ggml/src/ggml-cpu/ggml-cpu.c:1311` and `:1387` dispatch through SGEMM before the stock dot-product fallback. Maximum rollback retains one token, so its independent reference crosses that dispatch boundary. Disabling this CPU arithmetic path removes the discrepancy without changing rollback logic. Reduction/activation rounding is the supported explanation; the first divergent tensor and its exact arithmetic subtype were not traced with a per-node f64 or all-F32-weight oracle.

D: `src/llama-memory-recurrent.cpp:195` sets the pending distance to `cell.pos-(p0-1)`. `:1363` addresses `idx*mem->size+src0`; the shared hybrid input fills this vector once at `src/llama-graph.cpp:1111`. The index is consumed once and reset before the corrected token completes.

D: CPU GDN writes slot `n_tokens-1-t` at `ggml/src/ggml-cpu/ops.cpp:11041`. For the retained token `t=keep-1`, this equals `r`. `src/models/delta-net-base.cpp:588` copies all produced slots to matching cache planes. Conv uses slot `K-t` and the corresponding input window at `:505`. A verify block contains `k+1` tokens and overwrites every required plane, so selecting maximum rollback cannot read an old prompt snapshot.

D: `src/llama-memory-recurrent.cpp:445` passes the trailing-window constraint to `split_equal`; `src/llama-batch.cpp:649` begins its enforcement. At prefix `257`, ubatch `128`, k `5`, the prompt split is `128,123,6`; prefix `129` splits `123,6`. The sweep covers these boundaries and the neighboring configured depths.

## Reproduction and provenance

Private shared-object clone: `/home/xanewok/repos/ra-k5-state/stack2`, branch `codex/k5-state`, inherited diagnostic tip `6827b95a3`. Production CPU/model/memory sources match `0dba27b90`, the supplied stack2 descendant of `68d99c67d`. The inherited CUDA-only diagnostic commits do not execute in these CPU tests. The original native build at `/home/xanewok/repos/ra-gpu-paths/stack2/build-cpu` and every external report/model/tree were read only.

CPU model SHA256 (M): `d2bfbee4de17c74e6308a4dc750be4c2271f92ae2df1e28f5c5afa0dcdb6fccc`. `state-inputs.sha256` fingerprints the tested libraries and sweep binary; `final-probe.sha256` fingerprints final CPU-linked probes. `state-evidence.json` retains per-case native/strict measurements and sweep aggregates. Raw logs remain in this worktree.

From the clone, full reproduction:

```sh
CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/run_state_cpu.sh
```

The launcher configures `cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CPU_REPACK=OFF -DGGML_LLAMAFILE=OFF`, disables ccache, CURL and UI download hooks, builds with `-j8`, sets temporary storage inside this worktree and runs contexts with at most eight compute threads. The native comparison uses the supplied read-only CPU libraries through `LD_LIBRARY_PATH`.

Minimal corner after building the probe:

```sh
CUDA_VISIBLE_DEVICES='' E_PREFIX=257 E_K=5 E_R=5 E_SERIALIZED=1 E_EXACT=1 nice -n 19 build-probe/state-split-probe /home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf
CUDA_VISIBLE_DEVICES='' E_PREFIX=257 E_K=5 E_R=5 LD_LIBRARY_PATH=/home/xanewok/repos/ra-gpu-paths/stack2/build-cpu/bin nice -n 19 build-probe/state-split-probe /home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf
```

M: CPU build, final probe compilation, parent-enabled probe compilation against CPU-only libraries, strict checks, original-fixture q8 controls, negative controls, shell syntax and GPU plan/authorization guard all pass their expected outcomes. No CUDA-linked executable, GPU query, server process or network operation was run. There is no CUDA source change requiring a new CUDA build.

Operational deviations: the initial filesystem-only `pwd`, skill read and directory listing omitted an explicit empty CUDA environment. Initial native repro and CPU build also overlapped; log timestamps bound that interval to `2026-10-01 09:19:51.708..09:20:08.235 CEST` (M), with a possible aggregate sixteen compute threads (D). This was a scheduling mistake. Subsequent heavy CPU commands were serialized. Parent should exclude that interval if it overlaps a host-sensitive GPU measurement.

## Parent-only GPU twin and remaining scope

`codex/gpu_ab.sh` defaults to plan mode. Its default compares prompt ubatch `128` versus `64` on frozen stack2 bytes, boots `up,patch,patch,up`, and runs the decisive prefix/k/distance with a teacher-forced continuation. It records raw full-vocabulary logits, conv/GDN errors and serialized-state equality. Optional cases isolate fusion, BF16 AllReduce or CUDA graph use. Repeated boots must have byte-identical artifacts; cross-arm differences are reported for diagnosis rather than judged by an arbitrary CPU tolerance.

Parent command, never run here:

```sh
RUN_PARENT_GPU=1 PLAN=0 nice -n 19 bash codex/gpu_ab.sh
```

The script freezes libraries, probe, input/source hashes and `SHA256SUMS`; checks matching internal C++ layouts against frozen source; refuses active devices/serving processes; checks Xid before and after each boot; bounds execution by `BUDGET_S` with bounded cleanup; and stops only its owned PID. Device/kernel-log queries are time-bounded. The default budget is `1200` seconds, prefix is `257`, k/distance are `5`, continuation is `200`, q8 KV and FA are on, tensor split is `50,50`, and the head is full. Prefix is capped at the probe's `2048`-token batch capacity. The parent-only split array satisfies `llama_max_devices()`, as required by `src/llama-model.cpp:1291`.

This resolves the CPU discrepancy. The Swift GPU repetition bug remains unresolved. The twin covers target state on short synthetic histories, not long-context serving, installed backend samplers, draft/subset bookkeeping, server checkpoint RNG or the speculative prefill hook. No GPU correctness or speed claim is made.
