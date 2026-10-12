# stack11-mmvq: MMVQ 56/55 port notes

Branch `topic/stack11-mmvq` on `a38787dfb` (topic/stack11-base, master `abee0c847` + series). Code-only port: nothing built or run.

## Commit map (stack10 `a588f2a44` -> stack11)

| stack10 | pkg | stack11 | subject |
|---|---|---|---|
| b28f11784 | 56/0002 | d33312e83 | cuda : bind MMVQ launch dimensions to kernel parameters |
| 311452e04 | 56/0003 | f8a664aa8 | cuda : use one-row MMVQ blocks for partial tails |
| 96217fbd5 | 56/0004 | b8918a0e9 | cuda : reject unavailable MMVQ parameter tables |
| cc0bfd1b4 (general part) | 55 | b7a0a5688 | cuda : pass the quant type to the MMVQ rows-per-block choice |
| 8a1001e26 + cc0bfd1b4 (table data) | 56/0001 + 55 | b2a10f8b7 | local: cuda : Blackwell MMVQ launch parameters for 5-8 columns |

## Adaptations

- Base: since stack10's base (98c695111, merge-base 050439614) master changed `mmvq.cu` only in 1623d8ce4 (Pascal MUL_MAT_ID batch). The GB10 table, `small_k` and `halve_iters` were already in stack10's base, so b28f11784 already carried them in bound form; nothing to re-derive.
- b28f11784 embedded the Blackwell entry of 8a1001e26 (it sat before it on stack10). Removed from 1-4: enum value, `calc_nwarps`/`calc_rows_per_block` rules, `mmvq_select_config` case; the test loops to `MMVQ_PARAMETERS_GB10`.
- b28f11784 test-backend-ops: the MMVQ loop (q4_K/q5_K/q6_K/iq4_xs/q8_0, m 5-7, n 5-8, k 512) sits after master's "SIMD K boundaries" block, as in pkg 56 apply.txt.
- 96217fbd5: conflict on the Blackwell case line; kept only `default: GGML_ABORT(...)`.
- cc0bfd1b4 split: the general part (`type` parameter of `calc_rows_per_block`, `GGML_UNUSED` until a table uses it; `mmvq_select_config` passes it; test bound `nr <= 8`; CUDA specialization counter) is commit 4. The Blackwell per-type data goes to the local commit.
- local: Blackwell enum, host `get_device_table_id(cc)` branch (`ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_BLACKWELL`), table rules and test expectations. `mmvq-params.h` and `test-mmvq-params.cpp` are byte-identical to stack10 a588f2a44.
- Final `mmvq.cu` vs stack10 a588f2a44 differs only by master's 1623d8ce4 hunks and stack10's q8-cache commit 30afc9758 (out of scope).

## Table comparison

`python3 codex/stack11-mmvq/compare_tables.py` (text checks + hand-encoded tables over 23 types, ncols 1-8, small_k/halve_iters at ncols 1, 8 arch classes; 1840 configurations):

- Port general tables (b7a0a5688) are textually identical to master's `calc_nwarps`/`calc_rows_per_block` (apart from the `__host__ __device__` qualifiers and the unused `type` parameter). Port `mmvq-params.h` (b2a10f8b7) equals stack10's.
- Every arch other than sm_120: port (nwarps, rows) == master. GB10 halve_iters, small_k rows = nwarps, Turing, RDNA and GCN unchanged.
- sm_120: port == stack10 after 1-5. Versus master: 92 differences, all at ncols 5-8 with small_k/halve_iters off (master GENERIC (2,2)): (1,4) for q4_K/iq4_xs n5-8, q5_K n6-8, q6_K n7; (2,6) for q6_K n8; (2,4) for all other types. ncols 1-4 unchanged. These are the intended Blackwell entries. No difference comes from small_k/halve_iters.
- Negative controls: dropping the GB10 halve_iters rule from the port model fails 9 GB10 cells; pointing the text check at the general-form commit fails both Blackwell text checks.

Intended launch changes from 0002/0003 on every arch (same as stack10, not table changes):
- the kernel's nwarps/rows come from the host table (`ggml_cuda_highest_compiled_arch`), not from `__CUDA_ARCH__`. The result is the same whenever both pick the same table (native builds).
- `nrows % rows` tail rows are a second launch with one row per block, instead of a bounds-checked partial block.

## GPU window

```
build/bin/test-mmvq-params
build/bin/test-backend-ops -b CUDA0 -o MUL_MAT -p '^type_a=(q8_0|q4_K|q5_K|q6_K|iq4_xs),type_b=f32,m=[0-9]+,n=[1-8],k='
```

The filter selects ncols 1-8: the m=16 all-types loop (n 1-8), the Q4_K/Q5_K 4096/1023-row cases, and the new m 5-7 x n 5-8 cases. On cc 12.0, q4_K > 5, q5_K > 6 and q6_K > 7 columns route to MMQ, so those Blackwell entries are not exercised.
