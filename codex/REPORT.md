# Draft attention cause investigation

Recommendation: drop both CUDA tuning commits; no measured speed gain. The f16 CPU reference loses accuracy over long KV sequences. Attribution of the original patch failure still needs identical-input GPU data. See [REPORT-cause.md](REPORT-cause.md) for evidence, limits and the per-commit verdict.

`ggml/src/ggml-cpu/ops.cpp` now accumulates V in f32 when `use_ref` is set; native math is unchanged. No sampling or CUDA kernel code changed. CPU results and negative control are in `cause-cpu-*.txt`; harness results are in `cause-probe-tests.txt`, `cause-preflight-results.txt`, `cause-gpuq-check.txt`.

## Reproduce CPU checks

Run under `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, and `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8`:

```
cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build build-cpu --target ggml -j8
python3 codex/draft-attn-cause-probe/prepare_driver.py
c++ -O3 -DN_THREADS=8 -std=c++17 -Iggml/include codex/draft-attn-cause-probe/cpu_cause.cpp -Lbuild-cpu/bin -Wl,-rpath,'$ORIGIN/../../build-cpu/bin' -lggml -lggml-base -lggml-cpu -pthread -o codex/draft-attn-cause-probe/cpu-cause
codex/draft-attn-cause-probe/cpu-cause 0 8
```

Reverting the `!params->use_ref` guard fails the regression at `reference_err < 1e-10`; the raw failure is in `cause-reference-mutant.txt`. The oracle negative control `cpu-cause 0 1 broken` must abort at `err > 1e-4` (disable core files first). It replaces the independent oracle with the same inaccurate half accumulator. The native half accumulator must cross the unchanged tolerance at seed 0/GQA=4; the fixed reference must match the decoded-f32 V oracle within 1e-10. `cause-cpu-results.txt` retains the pre-fix eight-seed sweep; `cause-cpu-positive-final.txt` checks the fix.

Build the shared host probe with `bash codex/draft-attn-cause-probe/build_driver.sh`. Run `python3 -m unittest discover -s codex/draft-attn-cause-probe -p test_probe.py`. The real GPU artifacts test ERR/SQL, geometry, device, memory and hazard parsing; oracle gate mutations use augmented recorded rows because no GPU oracle artifact exists yet. This part needs the real canary.

`gpuq submit --check "$PWD/codex/draft-attn-cause-probe/item.toml"` passes; nothing was submitted. `gpu_ab.sh` is the probe wrapper. Frozen libraries stay read-only in `/home/xanewok/repos/draft-attn/codex/frozen/`; their identity gate is unchanged. The driver hash and all runtime inputs are pinned. Canary: one paired suite on GPU0, no server; full: both GPUs and seeds 0/1/2 in ABBA order. All raw results are saved before numeric gates.
