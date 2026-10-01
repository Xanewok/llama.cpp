# Verify-batch logits on runaway histories

## Conclusion

GPU equivalence remains unmeasured. No GPU query, CUDA binary, server process, network operation or external-file write was executed. Production code is unchanged. The parent probe is ready; its default is plan mode and its total wall budget is capped at the literal setting `1800` seconds.

M: the CPU Qwen3.5 fixture is **not** equivalent between scalar decode and verify decode. With q8_0 KV, maximum absolute logit difference is M: 0.540583 and maximum filtered probability difference is D: 0.0350448. Independent contexts reproduce both paths bit-identically; a second process boot has M: zero logit noise and D: zero probability noise. This establishes a CPU decode-width discrepancy on this fixture, not its cause or a CUDA explanation for the campaign runaways.

Exact original histories cannot be recovered from this archive. Original token IDs, raw requests and tool schemas are missing. The diagnostic uses identical reconstructed token IDs across every arm, but these are **approximate campaign prefixes**. Do not label the runaways chance or the serving stack correct from this work.

Labels: M = measured/observed, D = derived from text or measurements, G = guess. Flags, source identifiers, configured settings and source line numbers are literal identifiers. No GPU speed or full-matrix completion-time estimate is claimed.

## Source and reconstruction

Local clone: `/home/xanewok/repos/ra-verify-rows/stack2`, branch `codex/verify-long-rows`, based on frozen stack2 `0dba27b9002b8347c6f14a9f4a1651b3478065ef`, a descendant of the requested stack base `68d99c67d`. The original weekend clone and frozen libraries were read-only. The supplied GPU sampling reports, attribution, Patch A report and review, CPU launcher, frozen queue script and prior runaway report were read before design. `SP` resolves directly to `scratchpad`; there is no extra `SP/` directory under it.

`reconstruct.py:61` reads only `response_item` entries, avoiding duplicated `event_msg` text. It retains ordered input messages, exact saved tool results, previous reasoning and tool arguments, plus `session_meta.base_instructions`. It converts truncated S5 JSON command text back to its unescaped string. The saved per-attempt template renders completed history using the **frozen server's Responses conversion and chat implementation** (`probe.cpp:183`). Current reasoning is appended without trimming; current tool XML is inferred from that template. `HISTORIES.json` records paths, original rollout lines, SHA256 hashes, settings and limitations.

The onset is the start of the longest qualifying letter or zero-row span, not a proven causal transition. `probe.cpp:199` tokenizes the whole rendered stream, then locates the longest common token prefix before the character marker. This handles a marker inside a BPE token without independently tokenizing and concatenating fragments. Initial S4 request calibration is D: 5,386 reconstructed prompt tokens versus M: 6,428 recorded prompt tokens, a D: 1,042-token deficit. Omitted tool schemas and their template header are a likely contributor; this is not proof that they account for the entire deficit.

| History | Saved event line | Prompt tokens (D) | Onset/normal point (D) | Total tokens (D) | Available depth points (D) |
|---|---:|---:|---:|---:|---|
| S4 alphametics pair1 | 26 | 7,482 | 14,750 | 101,354 | 14,750; 20,000; 50,000; 90,000 |
| S4 alphametics pair2 | 28 | 7,839 | 18,793 | 86,882 | 18,793; 20,000; 50,000 |
| S4 alphametics pair3 | 31 | 7,747 | 13,980 | 101,358 | 13,980; 20,000; 50,000; 90,000 |
| S5 nucleotide-codons | 34 | 7,712 | 10,246 | 101,353 | 10,246; 20,000; 50,000; 90,000 |
| S0 alphametics pair2 | 96 | 20,968 | 21,208 | 21,254 | 20,000; 21,208 |
| S0 doubly-linked-list pair3 | 226 | 48,033 | 48,566 | 48,608 | 20,000; 48,566 |

The onsets precede the requested long-context range. Later points use real saved continuation, without padding, concatenating sessions, or inventing tokens. Normal points are near the end of a saved non-runaway response. Retokenization can choose a different segmentation than generation; parser stripping, tool XML whitespace, input assembly and unarchived tools further limit fidelity.

## Probe contract

At depth `d`, prefill stops before anchor `d-1`. Row zero consumes the anchor and predicts saved token `d`; subsequent rows predict the next saved tokens. This avoids comparing off-by-one logits. Target and MTP catch-up prefill use the same teacher-forced tokens; MTP inputs pair token `x_i` with target hidden state `h_(i-1)`, starting with zero carry, matching `common/speculative.cpp:1581`. MTP draft rows then chain their own hidden states, matching `common/speculative.cpp:1782`.

`probe.cpp:101` saves recurrent partial state and trims appended attention KV after restoration. Each point captures scalar reference, restored scalar replay, verify widths corresponding to `k=3,4,5`, and MTP draft rows. Width sweeps share the **same** target prefix, context and snapshot layout. Parent defaults match each archived campaign's target snapshot count; draft snapshots are disabled and `ctx_other` is set (`probe.cpp:226`, `common/speculative.cpp:2622`). Context size, logical batch, microbatch and split settings match the archived campaign. The separate `RS=3/4/5` override permits a snapshot-layout sweep without expanding the default matrix.

Full F32 logits and positive filtered probabilities are retained. Target filters use the configured top-k/top-p/min-p/temperature chain; draft filters use the actual probabilistic MTP `top_k=10` plus temperature (`common/speculative.cpp:62`). `probe.cpp:141` requires backend candidate sampling to match full-row host probabilities and the same-seed draw. This is a sampler check, not a sampling change. Backend installation status is recorded; CPU runs installed both target and draft samplers.

`compare.py:36` reports per-row maximum absolute difference, raw and filtered forward KL, top-20 symmetric difference, filtered support changes, probability differences and total variation. Infinity is explicit when a reference-supported token disappears. Comparisons include verify versus that boot's scalar reference and configuration effects versus the tensor/q8_0/MMA-on/subset scalar or draft reference. Subset masking is expected; common finite draft logits are still compared. Repeated-boot noise includes both scalar and matching verify rows. Without both relevant independent boots, classification is unresolved; zero within-boot replay noise is not substituted for boot noise.

Each row also records M: 4,096 draws from its actual filtered probabilities with the fixed diagnostic seed, including M: 16 sample IDs. D: the next saved token's probability, the mass of the IDs in the next saved-token window, and teacher-forced path products quantify continuation differences. The ID-set mass is not the probability of an indefinitely repeating pattern. These are **ungrammared** distributions: original grammar state and server RNG seed were not archived, and S5's tool-argument grammar can change the serving law.

## CPU evidence

Reused `/home/xanewok/repos/s4-runaway/stack2/build-cpu`, verified `GGML_CUDA=OFF` and `GGML_CPU_REPACK=OFF`. Its differences from frozen stack2 are tests and diagnostic files, not production code. All executed model/probe runs used empty CUDA visibility, nice level `19`, and at most the literal setting `4` compute threads; compilation used a single compiler process. CPU fallback build uses `cmake -B build-cpu -DGGML_CUDA=OFF`, disables network/UI hooks and ccache, and builds with `-j8`.

Fixture: `/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`, short saved-text token prefix, depth settings `32,48`, target snapshot setting `5`, draft snapshots disabled. Fixture subset has M: 24,832 descending every-tenth IDs, not the Swift subset. Raw local evidence is in `build-cpu-contract/`, `build-cpu-noise/` and `build-cpu-rs{3,4}/`; the small numeric extract is committed as `EVIDENCE.json`.

| CPU configuration | Verify rows (M) | Max absolute logit diff (M) | Max filtered probability diff (D) | Support-changed rows (M) | Top-20-changed rows (M) | Max raw KL (D) | Max filtered TV (D) |
|---|---:|---:|---:|---:|---:|---:|---:|
| q8_0/full | 30 | 0.540583 | 0.0350448 | 1 | 4 | 0.00386033 | 0.0608338 |
| q8_0/subset | 30 | 0.540583 | 0.0350448 | 1 | 4 | 0.00386033 | 0.0608338 |
| f16/full | 30 | 0.595540 | 0.0481111 | 8 | 11 | 0.0155818 | 0.0878182 |
| f16/subset | 30 | 0.595540 | 0.0481111 | 8 | 11 | 0.0155818 | 0.0878182 |

M: each configuration's independent fresh contexts reproduce all 42 scalar/verify rows, laws and sampled IDs bit-identically. The same independent-context checks also pass with target snapshot settings `3,4`. M: all restored scalar replay rows equal the initial scalar rows. M: the additional q8_0 full-head process boot reproduces every scalar and verify row with zero logit noise; D: probability noise is zero. M: 15 distinct q8_0 verify rows exceed that noise, repeated across both boots. M: same-seed sampled IDs change in 10 q8_0 rows and 9 f16 rows. M: full/subset target logits are bit-identical, and selected draft logits have zero error across 10 draft rows per KV setting. The mismatch occurs at `k=3` as well; it does not establish a `k=4` padding-specific failure.

Positive validation does **not** assert scalar/batch equivalence: that oracle fails on this native CPU fixture. It asserts that each path agrees with an independent context, that raw/backend probability laws agree, and that probabilities normalize. The deliberate `VERIFY_SWAP_ROWS=1` mutation makes validation exit with `FAIL: independent context oracle failed` at `d32/verify-k3-r0.f32`. Synthetic comparator controls also detect changed logits, probabilities and support. This avoids a meaningless negative control against a baseline already failing the equivalence oracle.

Executed: `CUDA_VISIBLE_DEVICES='' OUT="$PWD/build-cpu-contract" nice -n 19 bash codex/run_cpu.sh`; shell syntax checks; plan mode; parent-flag rejection; compilation with `VERIFY_PARENT_GPU` against CPU libraries only. The parent-capable binary was **not executed**. Repeated-boot analysis is reproducible with `CUDA_VISIBLE_DEVICES='' nice -n 19 python3 codex/compare.py build-cpu-noise`. No CUDA build was needed because the probe links the already frozen stack libraries.

## Parent queue and remaining limits

`gpu_ab.sh` freezes libraries, copied probe source, probe binary and comparator with SHA256SUMS. It pins the model to the archived SHA256, freezes history files and subset IDs, verifies manifests before boots, refuses busy devices/serving processes, checks Xid, and stops only its tracked child PID. `RUN_PARENT_GPU=1 PLAN=0` is required. Preparation, model hashing, compilation, probes, cleanup and analysis share the wall budget; analysis time is reserved. It does not start a process named `llama-server`.

Queue from this clone: `RUN_PARENT_GPU=1 PLAN=0 nice -n 19 bash codex/gpu_ab.sh`. There is no invented calendar restriction; the parent must choose an idle queue slot. Matrix settings are tensor/layer, q8_0/f16 target KV, subset/full MTP head and MMA knob off/on. Within each split/KV/head group, boots are off/on/on/off. D: this schedules 32 boots and 19 history points per boot; completion within the budget is unknown. Budget expiry preserves complete rows and reports missing boots, missing rows and unresolved noise rather than a clean verdict. If the full matrix cannot finish, use its coverage report to split the remaining arms into later bounded queue items; do not treat partial coverage as equivalence.

Outstanding: execute the parent GPU probe, localize the CPU discrepancy separately, and capture original request payloads/token IDs in any new reproduction. Matching reconstructed histories would narrow the decode-width hypothesis, not rule out proposal-dependent target state, long generation/rollback schedules, parser/grammar effects, or the known padded-mask issue outside this probe's exact states. The checkpoint RNG bug from the prior investigation remains separate and unchanged.
