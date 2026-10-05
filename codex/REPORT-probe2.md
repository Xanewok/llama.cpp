# q8 probe v2

Ready to queue; GPU correctness and speed remain unverified. M = measured or recorded; D = derived; G = guess. Caps and case dimensions are not measurements.

`q8-probe2/probe.py:165` gates q8_0 correctness, reports f16 NMSE/tolerance/pass in `correctness.csv` and `SUMMARY.json`, and treats f16 arm ERR spread above 1e-6 as integrity failure (`:200`). Missing metrics and malformed results fail closed. M us/run and D unique-KV GB/s retain all cases.

The old frozen driver prints ERR only on failure and randomizes inputs. `prepare_driver.py:14` instruments a shared C++ host driver with fixed tensor/worker seeds, distinct K/V names and all-case ERR output. It loads the existing absolute-path freezes, pinned by hashes; kernels and tolerances are unchanged. Fixed inputs differ from the historical random canary.

M recorded f16 `D128/KV90112/GQA6` ERR = 0.002350528; D tolerance = 0.0022528. This is upstream `test_serialized::max_nmse_err()` (`tests/test-backend-ops.cpp:8725`), 5e-4 * max(1, KV/20000), not a Python-probe threshold. Native `test_flash_attn_ext` instead uses fixed 5e-4 (`:8021`), which is stricter. Thus this is an upstream serialized-case tolerance failure; whether it is a false positive or numerical defect is unknown. Arm equality is still pending GPU execution.

M: 11 tests pass against real artifacts, including q8 failure and f16 drift negative controls. CPU-only build and deterministic K/V/mask check pass (`q8-probe2/{tests,cpu-check}.txt`). Frozen preflight passes. No CUDA build or GPU execution.

`q8-probe2/check.txt`: `q8-1row-fa-ab2: valid` (rc 0). Budget cap 1500 s. Full ABBA on both GPUs; paired canary on GPU 0. G full/canary estimates: 720/180 s. D perf minimum: 48 s/arm (`test-backend-ops.cpp:1705`); perf cap raised to 90 s from the impossible 30 s. Budget exits save partial data.

Parent submit command:

`CUDA_VISIBLE_DEVICES='' nice -n 19 gpuq submit /home/xanewok/repos/q8-1row-probe2/codex/q8-probe2/item.toml`
