# Package 54 padding invariance

Conclusion: M all 72 CPU cases pass byte invariance, finiteness and the changing-input control. A CPU mask mutant fails the invariant. GPU padding correctness remains unknown; no CUDA binary or GPU command was run. No production kernel or sampler changed.

Labels: M = observed measurement/artifact; D = source deduction/count/arithmetic; G = estimate or hypothesis.

## Design and cases

`tests/test-backend-ops.cpp:test_flash_attn_ext_padding` extends the existing attention test and uses its parameter filter, backend selection and SQL result transport. `test_case::eval` becomes virtual so this case can compare repeated evaluations on one backend instead of gating a cross-backend numerical tolerance. It runs zero/garbage/zero/garbage on the same graph, then unmasks a dominant garbage row. Active inputs and graph dimensions stay fixed during the padding comparison. Output must remain byte-identical and finite; the control must change output by more than D 0.002. For exact multiples with no logical padding, the control replaces the last visible row. Garbage also fills unused physical cache cells beyond Npad, so these cases still probe illegal tail reads.

D Q f32 `[256,1,24,1]` and `[256,1,12,1]`; K/V q8_0, GQA 6; f16 mask zero through the visible prefix and `-INFINITY` after it; no sinks, bias or softcap. D visible counts are each of 8192, 32768, 90112 plus {-1,0,1,5,64}; Npad rounds upward to 4096. Additional cases use Q scale D 20, one visible key with Npad D 94208, and visible count D 8252 with Npad D 12288. Total D 72 cases = 2 head counts x 2 layouts x (15 boundaries + 3 controls/probes).

The brief's layout claim needs qualification. `src/llama-kv-cache.cpp:1379` builds a `[256,heads,cells,streams]` view with interleaved heads; `src/llama-graph.cpp:2725` permutes `{0,2,1,3}`. After permutation its head stride is one quantized row and its stream stride covers the full cache. Both layouts are tested: `head_major=0` matches that source view with capacity D 102400; `head_major=1` additionally has the requested D 102400-cell head stride. Neither substitutes packed Npad storage for the full cache allocation.

M the recorded 8k KLD maximum is at zero-based row 59: `codex/padinv-evidence/kld-max-row.json`. D the scored decode at position 8251 sees 8252 keys, padded to 12288 (`kld-window.cpp` in the identity2 tree submits each scored position as one token). This extent is included. It is a seam probe, not a reproduction of the original attention activations: the artifact stores output logits, not layer Q/K/V, and no stream-K root cause is established.

The CPU f32 reference uses the same deterministic q8 bytes, casts K/V to f32, and evaluates f32 attention. Both reference and local CPU attention use one compute thread. Outputs are stored with D 9 significant decimal digits, which round-trip finite f32 values. The window computes NMSE and maximum absolute error for on/CPU, off/CPU and on/off per case. CPU reference equality checks input consistency across arms; same-route repeat equality is reported without a gate. No numerical accuracy bound gates the verdict.

## Pre-registered GPU verdict

Write this rule before GPU execution: a finite, completed case with broken padding invariance in either route on either GPU means **`defect`**. Otherwise all required shapes must be supported, finite, complete and control-sensitive to report **`no padding defect`**. Missing cases, wrong route engagement, malformed results, void controls without an invariant failure, failed computation or non-finite output mean `invalid`. A scoped kernel fault means `hazard`; timeout means `budget`. A passing result excludes only the tested finite-padding defect, not other package 54 numerical or history defects.

`codex/padinv_summary.py:summarize` implements the rule and writes `summary.json` and `errors.tsv`. It does not infer correctness from NMSE. The CPU mutant sets every suffix mask entry to zero; M it exits 1 with `finite=1`, `invariant=0`, `control=1`. The summary self-test also mutates valid evidence to verify the `defect` verdict and rejects disabled route engagement and invalid evidence flags.

## Window ready for parent validation/submission

Manifest: `codex/bw-pkg54-padinv.toml`; entry: `codex/bw-pkg54-padinv.sh`; runner/analysis: `codex/padinv_summary.py:execute` / `summarize`. Filler, priority D 16, `host_sensitive=false`, CCX wrapper `/home/xanewok/repos/qwen38-eval-worktrees/work/aa-load/ccx01.sh`. No item was submitted, cancelled or reprioritized.

Each attempt exclusively creates `/mnt/scratch/pkg54-padinv/a{attempt}`. It builds only `test-backend-ops` with recipe 120a, CUDA D 13.2, NCCL off, local CCCL D 3.4.3, disconnected FetchContent, ccache on and `TMPDIR=/mnt/scratch/tmp-build`. Configure and compile each pass through `buildgate`; compile is D -j4 and nice D 19. Devices stay hidden during build. The binary and shared libraries are copied, assigned `$ORIGIN` RUNPATH, hashed, and checked before/between/after arms. The source commit and exact flags are saved.

The internal canary uses one smallest boundary case on CUDA0 with both routes, capped at D 600 s. The environment switch is static per process, so it requires two test-process starts rather than one server boot. It exercises the same SQL/evidence, coverage, route, finite/control, journal and verdict paths as the full run. Full execution follows only after canary success: on/off/off/on on CUDA0, then CUDA1, D 8 process starts and 576 case records. Every process has an exact recorded PID, bounded wait and TERM/KILL cleanup. No server or model is used. Kernel journal checks start at the test-stage cutoff, avoiding historical faults.

G `est_s=3600`: cold build 1500 + test/reference work 1800 + sealing/analysis 300. Basis: M local one-thread CPU suite 136.16 s; D eight full starts would be 1089.28 s at that CPU-suite duration, with G extra slack for GPU initialization, journaling and different backend costs. This is a scheduling estimate, not a GPU speed claim. D `budget_s=7200`, split into build 3000, canary 600, full 3300, reserve 300; G estimate is below D 0.7 x budget. The new CUDA compile and actual GPU canary remain for the parent.

## CPU evidence and commands

M CPU-only configure/build passed with `GGML_CUDA=OFF`, `GGML_OPENMP=OFF`, `GGML_CCACHE=ON`, Release, tests on, tools/examples off. Both used `CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1`, nice D 19, affinity `6-11,18-23`, and `buildgate`; compile used D -j4. Sandbox-local shared-cache access used `CCACHE_READONLY=1 CCACHE_TEMPDIR=$PWD/build-cpu`.

M the CPU suite passes D 72/72 cases, with M wall 136.16 s and M peak RSS 1,200,160 KiB. M maximum q8-to-f32-reference NMSE is 0.00015368687148309583 and maximum absolute error is 0.0194258988. All per-case values are in `codex/padinv-evidence/cpu-errors.tsv`; maxima and parser checks are in `parser-check.json`. Raw local outputs remain in the ignored `cpu.sql` / `mutant.sql`. No CPU model was loaded. This was a single-compute-thread, OpenMP-disabled check during the allowed acceptance item, not a multi-core timing job; no `cpugate` gate was bypassed.

Reproduction (all commands use the environment, nice and affinity above):

- `buildgate cmake -S . -B build-cpu -DGGML_CUDA=OFF -DGGML_OPENMP=OFF -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF`.
- `buildgate cmake --build build-cpu --target test-backend-ops -j4`.
- `build-cpu/bin/test-backend-ops test -b CPU -o FLASH_ATTN_EXT -p '^padding=1,' --output sql -j 1`.
- `python3 codex/padinv_mutant_check.py` builds/runs the CPU-only temporary mutant through buildgate, removes only its own temporary source, and writes `mutant-check.json`.
- `python3 codex/padinv_summary.py selftest codex/padinv-evidence`.
- `PYTHONPATH=/home/xanewok/repos/gpuq python3 -m gpuq.shlint codex/bw-pkg54-padinv.sh`; `bash -n codex/bw-pkg54-padinv.sh`; Python AST parse; `git diff --check`.

M real-artifact parser checks use the recorded restrict-probe CUDA0 attention SQL (D 9 rows), the actual identity2 stock canary log (M 64 matching grouped-q8 markers), and `orchestration/gpuq/fixtures/xid-lines.txt` (D 4 matching fault lines). The newly added evidence fields do not exist in old GPU artifacts; they are validated on actual new CPU output and CPU-derived summary controls, then checked by the required real GPU canary. These controls are not presented as GPU runs.

M window preflight passes. Both normal-environment and clean-environment `gpuq submit --check` return `bw-pkg54-padinv: valid`; outputs are retained in `padinv-evidence/submit-check.txt` and `submit-check-clean-env.txt`. The clean command uses `env -i HOME=$HOME PATH=/usr/local/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1` and absolute `/home/xanewok/repos/gpuq/bin/gpuq`. M `shlint`, Bash syntax, Python AST and diff whitespace checks pass. Test commit: `36182f7a9` (`tests : check flash attention padding invariance`).

## Limits and open items

`gpuq status --brief` cannot take the state lock inside the sandbox (read-only state directory); the read-only status snapshot showed `spec-tf-j5q` acceptance running. No queue state was changed. The named old SP reports and old queue script are absent at their supplied paths; the task-specific identity report, source, reference windows, KLD artifact and harness contract were available and read. The worktree index lists this tree without pins. Existing user changes to `codex/PROMPT.md` and `codex/PURPOSE` are preserved; preflight deliberately excludes only those briefing files from the tracked-diff check.

Open: parent CUDA build/canary/full window; GPU error table and verdict; layer activation capture if visible 8252 does not expose a defect. No finding here proves that the 8k KLD result is a masking bug.
