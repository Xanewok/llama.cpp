# np2 getter gate fix

PASS: the full Mac CPU gate completed in M 346 seconds on `5a855ecc320bde2e40dddc90ba9523ec3b5dcb1e`. Getter and gathered control logits are byte-identical on identical batches, both cache fills; all mutants were caught. Both window manifests validate. Production code and sampling are unchanged. The parent can requeue the window to capture current pins; nothing was submitted or cancelled here.

## Design and source evidence

The single-sequence comparison is dropped rather than assigned an unproved backend-wide tolerance. The existing scenarios and default by-id/gather structure checks remain.

- `tests/test-recurrent-state-rollback.cpp:349`: `test_getter_gate` runs the two-slot, swapped-cell, joint-start and extra-state scenarios.
- `tests/test-recurrent-state-rollback.cpp:459`: `--getter-logits FILE` records a fixed key (cache fill, scenario, phase, sequence, position, vocabulary size), then the complete vocabulary float row. Missing/non-finite logits fail.
- `tests/test-recurrent-state-rollback.cpp:469`, `:1013`: `--getter-reference FILE` requires exact keys and float bytes, complete rows and EOF after the last record.
- `tests/test-recurrent-state-rollback.cpp:484`: `--getter-gathered` checks the control's structure against each GDN node's actual `v->ne[3]`. It does not force a production path. Multi-sequence nodes must gather; single-sequence nodes must read by id. An outer phase can split into single-sequence ubatches.
- `codex/np2/cpu-gate.sh:113`, `:125`, `:127`: build/run new and the existing `n_seqs == 1` control; compare their complete logit artifacts with `cmp`. The normal gate must first reject that control for missing multi-sequence by-id reads.
- `codex/np2/cpu-gate.sh:157`, `:173`: the wrong-row mutant and restored build replay against the gathered reference inside the test. Extra-state and tail-gather structure mutants, and plain/Meta driver comparisons, remain mandatory.

The existing `offload_kqv=false` option also changes cache/attention placement (`src/llama-model.cpp`, `src/llama-graph.cpp:2813`), so it cannot isolate the getter across backends. This uses the task's permitted two-build fallback, without a production hook or new root-series environment switch. A standalone getter run without a reference checks structure and finite logits; the window gate enforces exactness. The preferred single-binary comparison remains unimplemented.

The logit format is native-endian and for comparisons on the same host/configuration. The old `gate-diag.sh` is a historical diagnostic for the previous single-sequence test and is not part of this window.

## Commits

- `96774329b`: `tests : compare np2 getter logits on identical batches`; test/harness change, missing manifest pin, generated Mac-output ignore, and the pre-existing task brief edit.
- `5a855ecc3`: `tests : check getter control against ubatch sequence counts`; correction exposed by the Mac control run.

The final report commit changes documentation only. Prior untracked Mac outputs remain locally under ignored `codex/np2/mac-out/`.

## CPU evidence

Command:

```sh
CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 BW_MAC_OUTPUT_BASE="$PWD/build-bw-mac-evidence" /home/xanewok/repos/qwen38-eval-worktrees/bin/bw-mac "$PWD" codex/np2/cpu-gate.sh np2-gate-fix
```

Successful evidence: `build-bw-mac-evidence/np2-gate-fix-mac-a3/`, especially `summary.txt`, `runs.tsv`, `payload.log`, `mac-run.txt`, `gate-new.bin`, `gate-ctl.bin`, `mut-row.log`, and `gate-restored.log`. Remote copy: `/Users/builder/qwen38-bw/np2-gate-fix/5a855ecc320bde2e40dddc90ba9523ec3b5dcb1e/run.BMqPHt/out/`.

| Check | Result |
|---|---|
| Complete CPU gate | M rc 0; 2026-10-08 04:57:48 to 05:03:34 CEST; M 346 seconds, including builds |
| Getter/control vocabulary logits | M 124 rows x M 248320 vocabulary entries, fills `0` and `0x3e`; byte-identical; M 123169696 bytes per artifact |
| Full rollback suite | M rc 0 |
| Plain and Meta driver identity | M 288 tokens per mode and all draft counters identical |
| Normal gate on control | M rc 1; missing by-id multi-sequence nodes detected |
| Extra-state, tail-gather, wrong-row mutants | M rc 1 each; engagement and expected failure signatures present |
| Wrong-row exact comparison | M failure at `two slots: verify batch, both seqs`, sequence 1, position 19 |
| Wrong-row driver | M rc 0, output differs from control; mutant caught by comparison |
| Restored build | M rc 0; exact gathered-reference replay and clean remote tracked tree |

The supplied nice-19 Mac gates used at most three build jobs and four test threads, explicit small contexts, CUDA/Metal/BLAS/Accelerate off, repack/llamafile off and arm64 i8mm off. `cpu-gate.sh:7` bounds the entire Mac payload to sixty minutes, plus thirty seconds for termination. `cpu-gate.sh:54` waits at most ten minutes for at least four GiB available and no known model process, including before Mac builds.

Earlier attempts are retained: `mac-a1` skipped with M rc 75 because another model job occupied the Mac for the entire M ten-minute wait; no compilation/tests. `mac-a2` passed the new scenarios, full suite and both drivers, then failed with M rc 24 because the control checker assumed the joint-start prefill was a multi-sequence ubatch. The node-shape correction above fixed that assumption; the full rerun passed.

Static checks passed: `bash -n codex/np2/cpu-gate.sh` and `git diff --check`. No Linux compilation, tests, model execution or GPU command was run.

## Window and remaining work

After the Mac PASS, both commands returned M rc 0 with `valid` and no contract violations:

```sh
CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 gpuq submit --check /home/xanewok/repos/qwen38-eval-worktrees/wt/np2-kernels/codex/np2/bw-np2-getter.toml
CUDA_VISIBLE_DEVICES='' GIT_NO_LAZY_FETCH=1 PYTHONDONTWRITEBYTECODE=1 gpuq submit --check /home/xanewok/repos/qwen38-eval-worktrees/work/stack7-items/bw-np2-getter-ccx01.toml
```

The local manifest lacked the transitive `work/build-window-1007/mmq.sh` pin; `codex/np2/bw-np2-getter.toml:36` now includes it. The parent's copy already has it. No new comparison helper/pin is needed. GPU scripts, templates and parsers are unchanged.

The new raw-logit gate remains unrun on x86 here. The recorded x86 diagnostic supports the batch-shape diagnosis, but identical rounded NMSE diagnostics do not establish raw-logit byte identity. The Linux build window must pass this exact comparison before freezing. CUDA kernel verification and GPU performance remain the parent's window/items work; no GPU speed result is claimed.
