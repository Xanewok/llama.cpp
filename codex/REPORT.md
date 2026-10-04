# Leading logit bias before backend top-k

Labels: M measured/observed; D derived; G prediction. Configuration values and identifiers are literal.

## Conclusion and design

`eeb3c2f11` fixes the prefix; `bda9338d8` tests it; `1764a4302` makes duplicate fallback explicit. No model-specific or local variant is needed. Candidates accept leading finite or `-INF` logit bias.

`common/sampling.cpp:580,611` installs a separate `[logit_bias, top_k(K+1)]` backend chain. Backend bias scatters into a replicated vocabulary table and maps it through global candidate ids (`src/llama-sampler.cpp:3999`). Meta `GET_ROWS` permits replicated tables with split global indices (`ggml/src/ggml-backend-meta.cpp:940`). Only sparse values/ids are uploaded; the backend builds the dense table.

`llama_sampler_chain_leading_top_k` accepts empty samplers and valid pointwise biases before top-k (`src/llama-sampler.cpp:4413`). The TP gate uses this helper (`src/llama-context.cpp:1355`); the graph applies the prefix before shard top-k and skips it after the exact gather (`src/llama-graph.cpp:3948,3985`). Unbiased chains retain a shared gather.

Exactness: if a token belongs to the global biased top-K, at most K-1 tokens rank ahead of it in its own shard. Therefore the union of shard top-K contains the global biased top-K. The existing finite, strictly decreasing K+1 certificate (`common/sampling.cpp:158`) admits only a uniquely ordered top-K. `apply_chain` runs the original host suffix/RNG without adding bias again (`common/sampling.cpp:199`). Fallback runs the original chain on raw logits. No bias subtraction or multiplication of `-INF` occurs.

## Fallbacks

Full-row work remains for active penalties/DRY or another distribution-reading sampler before top-k, unsupported top-k sizes, duplicate bias ids, tie/nonfinite certificates, grammar-first, grammar rejection, and reasoning-budget forcing (`common/sampling.cpp:286,577,833,888`). Duplicate or invalid ids are refused by the TP prefix helper; duplicate bias samplers remain on the CPU because scatter overwrite cannot reproduce CPU duplicate additions. Common tables exclude invalid ids, matching CPU behavior. Existing TP gates remain.

## CPU evidence

Commands used `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, at most eight threads. Qwen3.5-4B from `SP/patchA/run_cpu.sh` is absent; tests used recorded `SP/models/stories15M-q4_0.gguf` and the generated TP model.

Commands:

- `cmake -B build-cpu -DGGML_CUDA=OFF`
- `cmake --build build-cpu --target test-sampling test-sampling-prefilter test-backend-sampler test-llama-archs -j8` (worktree CCACHE_DIR)
- `build-cpu/bin/test-sampling`
- `python3 codex/run_sampler_cases.py`
- `build-cpu/bin/test-llama-archs --tp-sampling -s 42`
- `python3 codex/check_bias_controls.py`

M: 3,000 seed comparisons pass: two biases x 300 seeds x (one CPU row + four TP rows), with identical tokens/logits/probabilities. Finite biases promote tokens outside raw top-K in both shards. M: twenty fallback comparisons pass (tie, nonfinite, grammar-first, grammar retry, forcing). Deferred raw rows compare byte-for-byte. See `tests/test-sampling-bias.h:38,69,95` and `codex/bias-tp.log`.

M: `test-sampling` passes, including 24 queue cases (`codex/sampling.log`). Backend sampler sweep: 21/23 pass, or 20/22 existing cases plus the new case (`codex/sampler-cases.json`). `logit_bias` and `penalties` abort identically on base `98c695111`; the former assumes a stochastic biased draw must select its chosen token, the latter fails its masked-logit assertion. Base also fails the new regression. Existing TP checks pass. `test-sampling-prefilter` builds but its Qwen/tool-template suite could not run without the missing model.

M: both production-source negative controls fail the new regression: double bias and selection of raw candidates followed by host bias. The driver restores/rebuilds on exit; results are in `codex/controls.json`. M: restored run costs 0.286 s CPU and 14.465 s TP, including setup/distribution tests.

## Parent GPU A/B and freeze

`codex/gpu_ab.sh --self-test` passes real response, memory, fault-log, Prometheus and frozen-metadata artifacts (`codex/gpu-parser-tests.log`). `--run-parent --canary` runs one boot, warmup and each case plus a repeated depth request for the identity gate; budget capped at 600 s. Full mode runs the canary, then up/patch/patch/up. It saves every request before gates, compares explicit token ids, uses authoritative draft-round deltas, waits for idle memory/RAM, checks faults since a cutoff, and stops only its exact child PID. G: stages total 1,695 s within 2,500 s budget. Requires `--up` frozen at `98c695111`; inherited stack3 is older. Parent: `codex/gpu_ab.sh --run-parent --up <base-freeze> --out <fresh-dir> --since <cutoff>` with two selected GPU UUIDs.

M: CUDA build and frozen manifests pass. `codex/frozen/leading-bias` records code commit `1764a4302`; RUNPATH uses `$ORIGIN` and CUDA 13.2, with no build-directory dependency. `codex/freeze_leading_bias.sh` checks RAM; builds CUDA 13.2, arch 120, NCCL off, -j6 without a device probe; freezes renamed binary/SOURCE/SHA256SUMS. M: final build start MemAvailable 22,458,848 KiB. No GPU runtime or benchmark was executed.

## Prediction, packaging and risks

G: harness depth requests move to the production candidates path, about -0.4 to -0.5 ms/round; zero expected on Codex traffic already using unbiased candidates. Added dense bias-table construction and per-biased-sampler gathers may offset some saving; GPU A/B decides. Multi-sequence CUDA execution, graph capture and the real Qwen MTP model remain unverified. Chain-only apply timing omits the biased suffix; common wall timing includes it.

Package placement (EVIDENCE headers): TP prefix/gather and id mapping extend `llama.cpp/04-tensor-backend-sampling`; common certified biased candidates belong with `16-backend-verify-sampling`, reusing `12-sampler-topk-prefilter` exact tie fallback. No upstream submission was created.
