# Mac build-window runner

The runner transfers a committed HEAD and executes the CPU payload on `mini-builder`. CPU build, both unit seeds and the code mutant have passed; fixture validation is still waiting for the Mac RAM gate. No local build, test binary, model process or GPU command was run.

## Interface

`CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 codex/bw-mac.sh "$PWD" codex/bw-cand-neginf.sh cand-neginf`

`TREE` must be a Git root with no tracked changes. `PAYLOAD` is a committed, tree-relative Bash script that takes an absolute output directory as its only argument; an absolute path under TREE is also accepted. `OUT_NAME` defaults to TREE's basename and accepts letters, digits, dots, underscores and hyphens. The reviewed script can be installed as `bin/bw-mac`; this task does not install it.

The Mac receives a HEAD snapshot bundle under `~/qwen38-bw/<name>/<HEAD>/run.XXXXXX/`, imports it into `~/qwen38-bw/objects.git` and creates a fresh checkout. The separate bare export cuts history at HEAD so missing historical blobs in the Linux partial clone do not trigger network fetches. The Mac object cache persists across bundles; this is not a guarantee of incremental transfer size. Execution is detached from SSH, and the local wrapper polls its exact PID and completion file.

Completed output is copied into `~/repos/qwen38-eval-worktrees/work/build-window-1007/<name>-mac-a<N>/`, atomically reserving the next free N. `mac-run.txt` records HEAD, hostname, start/end timestamps, contract environment and payload rc. `setup.log`, `payload.log`, `transport.txt` and all payload outputs are retained. `stage-wall-seconds.tsv` contains successful CPU-stage wall times, including any gate waits.

Exit code is the payload's code. For cand-neginf CPU stages: `0` pass, `2` unit failure, `3` mutant escaped or failed for the wrong reason, `4` identity failure, `5` path-proof failure, `8` build/run/preflight failure. Runner/setup/transport failures return `125`.

## Environment

| Variable | Linux payload default | Mac runner value |
| --- | --- | --- |
| `BW_CPU_ONLY` | `0` | `1`; successful stage 1 exits before CUDA/freeze/item work |
| `BW_MODEL` | `/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf` | `~/qwen38-stack7/models/Qwen3.5-4B-UD-Q4_K_XL.gguf` |
| `BW_GATE`, `BW_BUILDGATE` | `bin/cpugate`, `bin/buildgate` | Per-run pass-through `nice -n 10` script |
| `BW_JOBS` | `4` | `5` |
| `BW_CCACHE` | `ON` | `OFF`; ccache is absent on this Mac |
| `BW_CPU_EXTRA` | Empty | `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod -DGGML_METAL=OFF -DGGML_BLAS=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON` |

The runner sets `CMAKE_GENERATOR=Ninja`, prepends `~/qwen38-tools/bin` and `/opt/homebrew/bin` to PATH, and sets OMP/OpenBLAS/MKL/Accelerate thread limits to four. `CUDA_VISIBLE_DEVICES=''`, `GIT_NO_LAZY_FETCH=1`, `PYTHONDONTWRITEBYTECODE=1` and `GPUQ_BUILD_WINDOW=1` are recorded. The driver keeps `n_ctx=4096` and limits both plain and Meta CPU model contexts to four threads in CPU-only mode. Darwin model starts wait for no detected model process and at least ten GiB available, estimated from `memory_pressure -Q` and `hw.memsize` using the existing Mac gate convention.

## Changes and validation

- `codex/bw-mac.sh:19`: snapshot/object-cache import and isolated checkout; `:35` host environment; `:46` payload status; `:83` output reservation/collection.
- `codex/bw-cand-neginf.sh:59`: BW overrides; `:65` stage timers; `:131` Ninja mutant link; `:170` model-start gate; `:220` CPU-only exit. Default Linux configure/build/test/driver commands and parameters are retained.
- `codex/bw-cand-driver.cpp:45`: CPU-only thread limit; `:118` uses `json::array({p.tok, p.prob})`, required by the existing `common_json` API.
- `tools/server/server-context.cpp:1930`: explicit `size_t` lambda return; Darwin otherwise deduces different types for `pos + int64_t` and `pos`. Both results have the same width on the Linux and Mac targets.

`bash -n codex/bw-mac.sh codex/bw-cand-neginf.sh` and `git diff --check` pass. Linux execution equivalence was reviewed from the diff; local runtime validation is prohibited by the brief.

Mac source `b3e2874d0e01cc3b1513eb10f1fcf8314cb4f67e`, attempt `cand-neginf-mac-a3`, remote `~/qwen38-bw/cand-neginf/b3e2874d0e01cc3b1513eb10f1fcf8314cb4f67e/run.Mgs8C7/`:

| CPU stage | Result | Wall seconds |
| --- | --- | --- |
| Configure/build, tests enabled, driver link | PASS | 76 (M) |
| `test-sampling`; `--tp-sampling -s 42` and `-s 1729` | PASS | 26 (M) |
| Compile/link/run code mutant without `n_ban` extension | PASS: mutant caught | 2 (M) |
| Fixture driver matrix and comparer | Pending RAM gate | Unknown |

Negative control: mutant rc `1` (M), with `banned: read the full row although the top k is unique: row 0: id 9805, logit 0.000000, want 1.000000`. Both seeded units also report the unextended-chain control caught. Earlier attempt `cand-neginf-mac-a2` records and returns payload rc `8`, with failed driver compilation, proving nonzero payload status survives collection.

The completed logs, stage times and mutant artifacts were copied to `/home/xanewok/repos/qwen38-eval-worktrees/work/build-window-1007/cand-neginf-mac-a3/` while the detached fixture stage waits. `mac-run.txt` intentionally has no END/EXIT_CODE yet. The final source adds only the report, a Git-root check and automatic-maintenance disablement after the tested snapshot; the tested CPU payload and driver are unchanged.

## Window author requirements

Accept OUT as the sole argument, keep outputs there, and honor all BW host overrides without changing Linux defaults.
Exit successfully after CPU stages when `BW_CPU_ONLY=1`; CPU preflight must not require CUDA, gpuq, frozen binaries or Linux-only paths.
Support Ninja and Darwin paths/linking; use existing portable JSON/CMake APIs and pass `BW_CPU_EXTRA` to configuration.
Set explicit small model contexts, at most four model threads, and a model-start RAM/process gate; do not assume Git history before HEAD is available.

## Known gaps

Fixture validation remains unfinished while the Mac is below the RAM threshold; no end-to-end identity or path-proof verdict is claimed. Process detection is a pre-start check, not an atomic lease with unrelated Mac tasks; simultaneous external starts can race it. Only the named stack model executables are detected. Historical revision archives, additional fixture paths and arbitrary window-specific overrides need payload adaptation. Remote checkouts/bundles and local build-bw-mac scratch directories are retained; transport interruption leaves the detached payload running and collection must be resumed manually.

To finish validation, free Mac RAM without overlapping another model run. The detached payload will resume automatically; wait for `~/qwen38-bw/cand-neginf/b3e2874d0e01cc3b1513eb10f1fcf8314cb4f67e/run.Mgs8C7/exit-code`, then copy `out/.` back to the a3 directory if the local wrapper has exited. Record the final comparer verdict and `driver-and-compare` time in this report. `bin/`, the queued cand-neginf-bias tree, gpuq and the existing Mac stack directories were not modified.
