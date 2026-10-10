# Request-history tensor diagnostic

## Conclusion

No GPU root cause is established by this worker. The parent-facing item is `codex/history-root/history-root.gpuq.toml`; its dedicated entry is `codex/history-root/gpu_ab.sh`. The shared `codex/gpu_ab.sh` also dispatches its `preflight|canary|run` modes there. The manifest uses the dedicated entry because pin discovery through the shared wrapper includes unrelated legacy scripts. It replays fixed tokens in persistent contexts and reports the first observed tensor difference, target-logit differences, and repeat equality. No inference, sampler, server, or backend source is changed.

The supplied SP reports, attribution note, patchA review, CPU script, and old queue script are absent at their exact supplied paths (M). The retained `codex/REPORT-s10-identity2.md`, source, frozen provenance, harness contract, and recorded GPU fixtures were available. The vanilla result is now present: `/home/xanewok/repos/qwen38-eval-worktrees/wt/vanilla-master/codex/results/vanilla-history-iq4/full-a1/history-diagnostic.json` reports order dependence and unequal tokens/draft counters (M), with first differing token index 154 (M; source notes 2026-10-10 §19:2x, `history-diagnostic.json`). A fork-only explanation is insufficient; this does not distinguish stale state from arithmetic drift.

## Design and scope

`codex/history-root/history-root.cpp:319` reads a bounded HRT1 request sequence. Each process loads one target model/context and, except arm d, one same-model MTP draft context. Contexts survive every request in that process. The no-clear boundary removes sequence zero, as a cache_n=0 server request must, rather than calling `llama_memory_clear`; b and c instead clear target and draft memory with the respective data flag (`history-root.cpp:418`). Host `previous_h` survives boundaries, following `common/speculative.cpp:1690` and `:1782`; `common_speculative_begin` at `:1607` does not reset pending hidden state. Memory-clear arms do not add an unrelated host-state reset.

`codex/history-root/item.py:44` uses the pinned KLDT corpus and existing `harness.depth.corpus_prompt`. Short is D [8192, 90112]; long is D [8192, 32768, 90112]. Each request has D four fixed target decode inputs and D five additional fixed tokens available to draft steps. Both orders have byte-identical final requests. The driver runs separate catch-up when a target batch does not fold, then D five teacher-forced draft steps after each target decode and removes the speculative draft tail. It does not call a sampler or perform rejection verification. There are no sampled trajectories to compare or altered sampling laws to defend.

The archived history experiment requires `MTP chain: k=5` (`s10-history-iq4.toml`, stock_long canary gate, M). The input header therefore selects D k=5 and its rollback reservation, matching this failing profile rather than the opening project description's k=3.

The GPU command requests D context 102400, ubatch 512, batch 2048, one sequence, q8_0 target KV, f16 draft KV, flash attention, tensor split 50,50, KV padding 4096, input staging, and the queue-pinned reduced draft vocabulary. Target MTP weights, nextn output, and rollback configuration stay enabled in d to isolate removal of the draft context. This is a lifecycle approximation with one-row target decoding, not the served four-row verify graph or device draft chain.

For the final request only, `callback` records every scheduler graph-node output during the last prefill batch and the first D four one-token target decodes, including executed catch-up/fold and forced draft nodes. Domain identifies the executing context; folded MTP nodes consequently appear in `target`. Records contain domain, phase, graph-order index, name, op, type, shape, strides, op parameters, split axis, and per-device logical-byte SHA256, byte sum and byte absmax. Scalar F32/F16/BF16/I32 records also contain finite sum, absmax, min/max, and nonfinite count. Stride gaps are excluded from the hash. Target full-vocabulary logits are saved separately in the bounded KLDL format. Raw node tensors are never saved.

Inputs are read before the node executes and after a backend fence. Immutable model weights are identified by metadata and queue-pinned model bytes rather than copied on every node. A source whose private map/cache is not current is marked `unavailable_before_plan`; it cannot support an identical-input inference. Outputs must all be available. The last prefill batch is the first *observed* interval; an earlier prefill operation may have caused its state difference.

## Callback hazards and controls

Observed source facts: `ggml/src/ggml-backend.cpp:2081` splits execution at every requested callback node; `:2333` disables cached meta plans when a callback exists. `src/llama-context.cpp:1637` also changes the shared output scheduler choice. Callback tracing changes allocation, fusion, CUDA capture, and arithmetic partitioning. An untraced baseline is therefore part of the plan.

More critically, `ggml/src/ggml-backend-meta.cpp:2856` only reduces a PARTIAL subgraph when another subgraph follows it. A bare single-node callback can omit the required AllReduce. The tool interposes `ggml_backend_graph_compute_async` only while observing one node and appends a synthetic identity view, so a PARTIAL tail has a consumer and completes its reduction before the next callback graph assumes synchronized inputs (`history-root.cpp:30`). The interposition is explicitly exported and delegated directly to the existing backend interface. It also fences the backend before reading source fingerprints. Frozen libraries are neither replaced nor edited.

The meta getter rejects PARTIAL tensors and noncontiguous split views (`ggml-backend-meta.cpp:1744`, `:1835`). The diagnostic reads the actual simple-device slices instead. `meta_header.py` extracts the complete private tensor-map and buffer-context definitions from each pinned commit, without editing them; CMake records their SHA256. This is private-ABI diagnostic code for these exact frozen libraries, not a public extension API or an upstream patch. The CPU meta control compares a full graph, broken bare node graphs, and completed node graphs. M results are respectively 32, 16, and 32 (`evidence/fingerprint-test.log`); removing completion fails the correct-sum contract. Strided packing, transpose packing, ignored gap bytes, and active-byte mutation controls also pass (M). All three extracted private layouts have M SHA256 `e375272bbd87d21cc28f23130e5b07b8002773d47d65f2d84847006420541c4f`.

## Arms and pre-registered reading

Every included configuration runs both orders twice, with the second traversal reversed: D 28 fresh processes. The untraced arm has no callback, input fence hook, or synthetic completion views.

| Arm | Frozen engine and difference | Discriminates |
| --- | --- | --- |
| a | stack9-p55, sequence removal, target plus draft | Instrumented baseline |
| b | stack9-p55, `llama_memory_clear(mem, true)` between requests | Memory bytes plus metadata reset |
| c | stack9-p55, `llama_memory_clear(mem, false)` between requests | Metadata reset without wiping bytes |
| d | stack9-p55, target context only | Draft state and draft allocation/scheduling interactions |
| e | a with `GGML_CUDA_DISABLE_GRAPHS=1` | CUDA capture state under tracing |
| f | a on stack10 | Whether the observed difference survives on the later stack |
| untraced | a without `cb_eval` | Whether tracing removes target-logit history dependence |

Arm g is omitted: frozen vanilla lacks `llama_mtp_fold_init`, `llama_mtp_fold_arm`, and `llama_mtp_fold_n_last` (M symbol/header inspection); its private meta container and common-parameter layouts also differ. Linking the current headers to that library would not reproduce a safely comparable a arm. A separate native vanilla driver is needed.

1. Same-order repeats must match all trace records and target-logit bits. A repeat failure makes the experiment invalid before attribution.
2. Target dependence removed by b but retained by c is the pre-registered stale-memory-byte bug class. Inspect the first affected reader and downstream arithmetic; masked KV bytes or inactive rank storage alone do not establish leakage.
3. Removal by d implicates MTP/draft state or its allocation/scheduling consequences, not necessarily the draft's numerical computation itself.
4. Removal by e implicates CUDA graph state in the instrumented execution and requires an untraced confirmation.
5. An input/state/view difference points to state or placement first. An arithmetic output difference with complete identical observed sources is a numerical-order candidate. It is not proof of benign behavior, since backend state and the instrumentation remain possible confounders.
6. If untraced target logits depend on order but traced logits do not, stop with an inconclusive instrumentation result. If only draft/catch-up tensors differ, report that explicitly rather than asserting changed target arithmetic.

`codex/history-root/analyze.py:81` returns the first difference in execution order and per domain, with the parsed layer and op. An unnamed node inherits D layer from its observed source graph; graph/layout changes are reported separately from byte changes. `first_produced_by_domain` also reports the first difference after storage/view operations, so inactive cache contents do not obscure the first arithmetic output. `item.py:111` applies the repeat gates and emits `SUMMARY.json`. It preserves raw boot artifacts before evaluating gates; an invalid run does not become a numerical verdict.

## GPU item and budget

D `kind=measure`, priority 23, `host_sensitive=false`, `retry=fresh`, CCX wrapper `work/aa-load/ccx01.sh`, no ports. Output is fresh `/mnt/scratch/work/history-root/a{attempt}/{canary,run}`. Tool compilation occurs inside the item, separately linked to frozen stack9-p55 and stack10 libraries, with CUDA devices hidden, shared ccache enabled, nice 19, affinity 6-11,18-23, and buildgate `-j4`.

D budget 7000 s: canary cap 300 s, full stage cap 6600 s, and 100 s of cleanup allowance. G `est_s=4500`: D 28 boots x G approximately 135 s, plus G approximately 720 s for compilation, canary, fences/readbacks/checks and transitions. The retained identity2 report gives M approximately 151 s for the three-depth served workload and M approximately 6 s for model/context boot; the fixed output here is much shorter, while fingerprint readback and hashing add uncertain cost. This is an execution-cost guess, not a GPU speed prediction. The estimate is D below 0.7 x budget; each boot is capped at D 240 s and the global deadline also applies. Large KV views and per-node copies remain the main budget risk.

The one-boot canary uses the smallest long sequence with D [128, 256, 512], D context 4096, and the same padding, ubatch, capture, forced-draft, file, provenance and journal paths. It must observe target q8_0 KV, target two-device slices, the completion hook, every capture phase, complete logits, and the comparison negative control. It runs the same real-artifact parser controls before full work. CUDA canary execution is deliberately left to the parent.

Per boot, the runner checks frozen bytes before and after execution, observes loaded library paths, records the exact child PID, stops only that child, saves return/timeout/provenance status, and checks the kernel journal from that boot's start. Idle memory waiting and fault handling reuse the pinned harness implementation. The runner refuses execution without `GPUQ_ACTIVE=1`; preflight always hides CUDA devices.

## Validation

Final validation results and command evidence are recorded under `codex/history-root/evidence/`. CPU raw model traces/logits stay in ignored `build-history-cpu/cpu-model`; the committed summary is small. All local commands hide CUDA devices and disable lazy git fetch. Gated compile/model validation is reproducible with `nice -n 19 taskset -c 6-11,18-23 bash codex/history-root/validate_cpu.sh`; every compile uses buildgate and D four workers, and every model process uses cpugate, D eight threads and D context 4096.

The CPU wrapper requires a prepared `build-cpu` engine and a fresh `build-history-cpu/cpu-model` output path. Engine preparation used `cmake -B build-cpu -DGGML_CUDA=OFF -DGGML_CCACHE=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF -DCMAKE_BUILD_TYPE=Release`, then `cmake --build build-cpu --target llama-common -j4`, both through nice, taskset, cpugate and buildgate. The saved `evidence/cpu-engine-cache.txt` records the effective CPU configuration.

CPU-only engine configuration/build and diagnostic linking against CPU, frozen stack9-p55, and frozen stack10 succeeded (M). Frozen-linked executables were inspected with `readelf`, never executed. The reference CPU model was `/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf`, with q8_0 target KV, f16 draft KV, MTP, flash attention and D KV padding 256. D seven serial model processes used short [32, 96] and long [32, 64, 96]: a both orders twice, and b/c/d long once each. M runtime range was 49.59-61.10 s per process including loading, execution and fingerprinting; these numbers do not predict GPU speed.

M a same-order repeats matched every trace record and logit bit. M all four target-logit rows matched between a orders, and between a long and b/c/d long; D KL divergence was zero. The first observed a-order difference was prefill target `cache_r_l0 (reshaped)`, `RESHAPE`, D layer 0, node index 3. The first arithmetic output difference was `SCALE` at node index 5: M all 24576 finite output values had min/max/sum/absmax zero and no nonfinite values, despite unequal byte hashes. D this is signed-zero drift after scaling stored recurrent-state bytes by zero, not evidence that those bytes reached target probabilities.

The first catch-up difference was prefill `norm-32`, `RMS_NORM`, D layer 32, node index 3; its `mtp_h_input` source differed. Draft differences also occurred, including D layer 32 `FLASH_ATTN_EXT` at draft-0 node index 31. Thus the CPU replay observes request-history differences in draft inputs while target logits remain exact. It does not establish the GPU root cause. Locations, hashes, scalar statistics, comparisons and runtimes are in `evidence/cpu-model-summary.json`.

The validation shell's final postlude failed after the completed model runs because the script was edited while bash was executing it. The saved `validation-run.log` preserves that failure; the current script passes `bash -n`. Summary copying, rendering, parser tests and queue checks were resumed separately. No model result is claimed from the failed postlude itself.

M six Python tests passed, including same-order mutation rejection, identical-input classification, missing-input refusal, layout/order rejection, final-request byte identity, arm environment checks, and pre-registered reading controls. M four distinct shell files passed `bash -n` and `python3 -m gpuq.shlint`. Both `CUDA_VISIBLE_DEVICES= GIT_NO_LAZY_FETCH=1 /home/xanewok/repos/gpuq/bin/gpuq submit --check codex/history-root/history-root.gpuq.toml` and a clean-environment check using the absolute manifest path returned `history-root: valid` (M). Only `--check` was used; no item was submitted. Commands and final statuses are in `evidence/{submit-check.log,submit-check-clean.log,shlint.log,python-tests-final.log,validation.json}`.

Recorded GPU parser inputs are the actual KLF `/mnt/scratch/work/kld-s9-gdn4/out/kld-s9-gdn4/a1/canary/canary.klf`, the archived IQ4 canary log, and `orchestration/gpuq/fixtures/{xid-lines.txt,nvidia-smi-query.csv}`. No native GPU callback-node artifact exists for this new tool yet. Preflight serializes actual recorded GPU logit bytes into the callback schema and tests its hashes/order/phase gates; that validates parsing and comparison, not native meta capture. The queued canary supplies the missing native GPU validation and uses its own trace for the mutation control.

An initial parser-only test accidentally ran during `vanilla-depth-iq4` (M test duration 1.874 s, log completed at M 2026-10-10 16:38:13 UTC, plus unmeasured import/setup time). This violated the requested CPU-test gate. Subsequent tests and builds used cpugate. The parent should review any depth timing overlapping that test interval; no claim that the overlap was harmless is made.

Open: native GPU canary/full results; earliest causal prefill operation if the last-batch snapshot is already different; served speculative/rejection replay if this fixed-prefix approximation does not reproduce the behavior; native vanilla integration; and confirmation of any numerical-order interpretation with a nonperturbing backend trace.
