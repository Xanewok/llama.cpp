# stack10 drop map (stack10b)

Branch `topic/stack10b`, base `def4a4e20` (= frozen/stack10-cand2 source). Every component is a set of general commits followed by one
`local:` switch commit (env switches, mutants, engagement logs). gdn4 and K9 are not in this stack. Exceptions: k10 has a trailing
general test commit `d1022e454`; k11a has a second general + `local:` pair (`ecb1f2524`, `330ab2cb8`, the state I/O sync fix from
`wt/s10-k11a`) and a third `local:` commit (`5f1825dfa`, query-based debug check: local `ggml_backend_query` hook in
ggml-backend/CUDA/meta/CPU); mfix sits on top of the stack. Drops match by subject, so the position does not matter.

## Components and commits

Hashes at `topic/stack10b` (wt/stack10b; stack10 `198cec743` with K9 dropped by the recipe below, then the mfix pair on top).

| component | what | commits (oldest first; last = `local:`) |
|---|---|---|
| p55 | Blackwell per-type MMVQ launch parameters | `b28f11784`, `311452e04`, `96217fbd5`, `cc0bfd1b4` |
| k7 | paired q/k RMS_NORM+SCALE | `20ac6ec57`, `a4a3a8c63` |
| k4 | GGML_OP_SSM_CONV_RS (CPU, CUDA, Meta) + llama graph use | `386282191`, `6ac5f5897`, `6565c43db` |
| mfix | Meta recreates every main subgraph after an arena reallocation (the K4 np2 crash; general, test in test-alloc) | `c5087a10f`, `c3d17d5ef` |
| k6a | shared q8_1 activation cache (<= 16 rows) | `30afc9758`, `b4efb51a0` |
| k6b | ADD+RMS_NORM fusion (<= 16 rows); needs k6a test helper | `1376b053b`, `31e27ee33` |
| k2 | shared flash-attention KV-max mask scan | `fac1254d2`, `d045e5d49` |
| k10 | batched top-k (`d1022e454`: test_top_k::err bounds-checks output indices, on top of the stack) | `4bf1c2ea3`, `9b2ea143f`, `d1022e454` |
| k8 | MMQ stream-k default-tile units + small-I tiles (L1 prerequisite included) | `015a0cbc7`, `0ab5b19e7`, `3f0814173`, `c5baada6d`, `9ccfb56e0`, `7491039b9` |
| k11a | getter synchronize() elision; state I/O and KV stream copies keep the real sync; query-based debug check | `50af425d0`, `a434dc77b`, `ecb1f2524`, `330ab2cb8`, `5f1825dfa` |
| k11b | Meta input staging, slowest link first | `d5b2af7c9`, `a8eed261a` |

K9 (MMQ PDL) is not in stack10b: dropped with `drop-component.sh k9` (MMQ codegen regression, notes 2026-10-09). Its stack10 hashes were
`cd72032e8`, `b33022516`, `c15713fb1`, `23abb0b21`, `1ae85cd15`; its rows are gone from `dropmap.tsv`.

Full subjects: `codex/stack10-drop/dropmap.tsv` (component TAB subject). The recipe matches commits by subject, so it keeps working
after rebases and hash changes.

## Dependency

- Keep `mfix` whenever `k4` stays: without it the K4 decode graphs (fewer nodes than the gathered prefill graphs, more Meta subgraphs)
  reach a freed subgraph at the first np2 joint replay after 8k prefills (stack10 crash, `work/k11a-np2-fail`). `mfix` alone is a
  general fix and can stay without `k4`.
- Dropping `k6a` also drops `k6b` (K6b reuses K6a's `fill_identity_data` test helper and shares the activation-cache scaffolding in
  `ggml-cuda.cu`). The script adds it automatically. Every other component can be dropped alone or in any combination.

## Recipe

```
cd <worktree>; git switch -c scratch/drop-X topic/stack10     # never on topic/stack10 itself
codex/stack10-drop/drop-component.sh k7,k4                    # comma list = any subset; optional 2nd arg = base (default def4a4e20)
codex/stack10-drop/check-drop.sh                              # host syntax check, see below
```

`drop-component.sh` runs `git rebase -i` with `drop` for the listed subjects, then resolves conflicts that come from additions sitting next to
the dropped lines with `resolve-drop-conflicts.py`:

- A region where ours is a pure deletion of base lines is merged as theirs minus those lines (K4/K7 chain calls, K2 `CMakeLists.txt`,
  K6a `common.cuh`/`ggml-cuda.cu`).
- mmq.cuh launch lines: dropping K9 keeps K8's `tiles` template argument on the `<<<>>>` launches (rule `kernel_launch_rule`); dropping K8
  strips `tiles` from K9's `ggml_cuda_kernel_launch` calls. Dropping both has no conflict.
- The identity-check chain at the end of `test_backend()` in `tests/test-backend-ops.cpp` is edited by K4/K6a/K6b/K7/K2; after the rebase
  the script rebuilds it from the `run_*_identity` functions the tree defines (`--normalize`) and commits
  `tests : canonical identity-check chain after dropping ...` if it changed. A conflict the resolver cannot classify stops the script (exit 1,
  rebase left in progress): resolve by hand or `git rebase --abort`.

`check-drop.sh` is the compile check of the files this host can compile: no conflict markers under ggml/src/common/tests, then
`g++ -fsyntax-only` / `gcc -fsyntax-only` of the test programs and the host sources the components touch (`tests/test-backend-ops.cpp`,
`test-fattn-kv-max-cache`, `test-meta-input-stage`, `test-mmvq-params`, `test-topk-batched`, `src/llama-*.cpp`, `src/models/*`,
`ggml-backend-meta.cpp`, `ggml-cpu/ops.cpp`, `ggml.c`). The `.cu`/`.cuh` files are NOT compiled here: leftover references in CUDA code
are caught only by the CUDA build in the window (`COMPONENTS=` selects the retained set, see the window).

## Checked (M, 2026-10-09, scratch branches in `wt/stack10-dropdry`)

- stack10b (wt/stack10b, on stack10 `198cec743`): `drop-component.sh k9` completes (mmq.cuh conflicts at two K8 commits resolved by
  the script), `check-drop.sh` passes; the diff to stack10 outside `codex/` is K9's only (`ggml-cuda/common.cuh`, `mmq.cuh`,
  `quantize.cu`, `tests/test-backend-ops.cpp`). The mfix pair was committed afterwards; its drop is not checked (it touches only
  `ggml-backend-meta.cpp` and `tests/test-alloc.cpp`, no other component edits those lines).
- Older stack10 checks (hashes before the K9 drop):

- Single drops of all 11 components: rebase completes, `check-drop.sh` passes (conflicts resolved automatically in k4, k7, k6a, k2, k8, k9).
- Subsets `k7,k4` `k2,k10` `k8,k9` `p55,k8` `k6a,k11b` `k4,k6b,k11a`: complete, `check-drop.sh` passes.
- All 11 dropped: tree equals `def4a4e20` outside `codex/`, except the 3-line `alpha_ok` form of the canonical identity chain in
  `tests/test-backend-ops.cpp` (semantically the same; alpha check now runs unconditionally).
- Not covered: CUDA files are inspected by diff only (`git diff topic/stack10 scratch/drop-k6a -- ggml/src/ggml-cuda/ggml-cuda.cu` removes
  exactly the q8 block); the mutant/gate scripts of a dropped component stay in `codex/<component>/` and must not be run.
