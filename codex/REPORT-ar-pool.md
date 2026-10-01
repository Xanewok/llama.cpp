# AR pool diagnostic - DO NOT MERGE

## Conclusion

Ready for the parent's post-weekend GPU A/B. Branch `topic/ar-pool-diag` starts at `topic/round-trace` (`2b645a655`); runtime commit `1efa25025` is frozen in `codex/frozen/ar-pool`. M: CUDA compilation and CPU identity checks passed. No GPU binary, GPU query, network request or process named `llama-server` was run. `qwen38-weekend` was not accessed.

M = measured/observed, D = derived, G = forecast. Hashes, source lines, configuration values and histogram boundaries are identifiers, not performance numbers.

## Design and evidence

`ggml/src/ggml-cuda/ar-pool.h:8` validates `GGML_CUDA_AR_POOL`: unset defaults to 2; decimal integers 2..16 are accepted; empty, malformed and out-of-range values abort initialization (`allreduce.cu:400`). Each pipeline retains its validated value. `p->pool_size` controls modulo/lap (`:362`), event creation/free (`:457`, `:567`), arrival allocation (`:489`) and small-path payload allocation (`:509`). Event-handle capacity is the common maximum (`:315`); inactive slots create no CUDA objects. Active capacity is logged (`:539`).

All existing synchronization and reduction operations remain. In particular, both ranks still execute `cudaEventSynchronize(ev_pool[i][slot].ker)` before slot reuse (`:370`); copy-engine fences, shared large buffers, tokens and reduction arithmetic are unchanged. No sampler or RNG code changed. The existing round-trace instrumentation is inherited; `codex/attrib.py:173` adds per-phase/device AR wait counts, duration percentiles and bins over the retained steady rounds. API calls that return quickly remain in the histogram; wait duration is not GPU idle.

Per pipeline, pool 2 -> 8 adds D12 MiB mapped payload, D6 KiB arrivals and D1572 CUDA event objects. Fixed host handle arrays have the same capacity in both arms. Allocations and warmup are outside the comparison requests.

## CPU and compile checks

`CUDA_VISIBLE_DEVICES="" nice -n 19 bash codex/run_pool_cpu.sh` passed. The reference is a local archive of `2b645a655`, built inside this worktree. The Qwen3.5-4B MTP fixture compares completion token IDs, final target logits and teacher-forced draft logits in single CPU and Meta(CPU,CPU). M6,953,160 bytes per mode match across reference, default, explicit pool 2, pool 8 and pool 8 with tracing. Hashes and times are in `codex/ar-pool-cpu-results/summary.json`; local trace/accounting evidence is in that directory.

| Mode | Reference wall s M | Default wall s M | Pool 2 wall s M | Pool 8 wall s M | Trace 8 wall s M |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 11.986207 | 10.173909 | 10.068514 | 10.188984 | 10.279105 |
| Meta(CPU,CPU) | 23.930754 | 13.421986 | 12.982817 | 13.718650 | 20.833683 |

These are functional-test wall times under variable host load; CPU runs do not exercise CUDA AR and establish no pool speed result. Token/logit byte perturbations were detected. `codex/test_ar_pool.py` passed parser/range checks, M8 hardcoded-use negative controls and comparison of CUDA API/kernel call order with the parent. `codex/test_attrib.py` passed schema/accounting/SQLite controls, histogram boundary checks and rejection of aggregated wait events. `codex/test_pool_harness.py` passed plan/gate, completion/counter, missing-boot and wrong/fallback-pool controls, authoritative period arithmetic and S3/K flag checks. The inherited `codex/test_harness.py` also passed. Python runs used `PYTHONDONTWRITEBYTECODE=1`; shell syntax and `git diff --check` passed.

CUDA compile command: `CUDA_VISIBLE_DEVICES="" nice -n 19 cmake -B build-cuda -DGGML_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DCMAKE_CUDA_COMPILER_FORCED=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`, followed by `CUDA_VISIBLE_DEVICES="" nice -n 19 cmake --build build-cuda --target llama-server -j8`. M: exit 0; logs `codex/ar-pool-{cmake,build}-cuda.log`. Generated build metadata identifies `1efa25025`. The frozen executable is named `round-trace`; manifests cover it and its shared libraries.

## Parent-only queue

`codex/pool_ab.sh` defaults to a read-only plan; `codex/pool-plan.json` records the plan. Execute only after the weekend chain releases the GPUs: `RUN_GPU=1 CUDA_VISIBLE_DEVICES=<pair UUIDs> K=3 codex/pool_ab.sh --execute`. This command was not run here.

The same frozen bytes serve env-only pools in order `2,8,8,2`, then a separate matched trace-on `2,8,8,2` series. S3-style flags, pinned template, vocabulary subset and seeded tool/think inputs come from the inherited harness; `K` defaults to 3. Trace captures use fixed matched request counts, require at least 40 retained rounds per case and use `attrib.py` with exported NVTX/CUPTI data. The harness checks frozen bytes, model SHA, input stability, active pool, completion messages, authoritative verification/draft/accepted counters, Xid, idle GPUs, port and process ownership. Stops retain the exact PID/start time and owned Nsight session. Memory is sampled throughout each boot; G2400 s is the wall budget, configurable with `--budget`.

`summary.txt`/`summary.json` report each boot's `predicted_ms / spec_decode_num_drafts_total` delta (`tools/server/server-task.cpp:1562` defines the verification counter), acceptance, within-arm boot variation, sampled peak memory, steady V-to-V period, AR wait histograms, sync time and paired-GPU kernel-boundary idle. Request period includes ramp/final rounds; the trace comparison supplies the steady mechanism evidence.

Falsification, printed in the summary: if waits fall substantially while V-to-V time and kernel-boundary idle do not improve beyond boot variation, slot waits primarily represent backpressure on active GPU work. Moving waits to `llama_synchronize` or retaining submission spacing demonstrates no whole-loop capture saving. If waits do not fall, effective look-ahead did not change and the diagnostic is inconclusive. Unchanged host gaps alone cannot falsify capture's separate launch/submission-overhead mechanism.

## Open evidence

GPU execution, completion/candidate/q identity, cross-device protocol behavior and Nsight export compatibility remain unverified. Sampled memory can miss transient peaks. Short tool completions can fail the steady-round minimum; increase the same `--capture-requests` for all arms rather than relaxing the minimum. No capture implementation, stale-epoch/early-overwrite GPU control, wait suppression or GPU speed claim is included.
