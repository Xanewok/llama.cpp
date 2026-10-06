# 67: GDN recurrent-state getter

## Design

Primary: optional `ids` on `ggml_gated_delta_net`, carried as `src[6]`, and a `build_rs` getter like Mamba. No new opcode. `ids == NULL` preserves the existing contiguous state input. CPU and CUDA select the input row; the output remains in sequence order. This changes the ggml C signature; callers and libraries must be rebuilt together.

The getter uses the existing tail gather and extra-state relocation. It has no `n_rs == n_seqs` branch, so reserve and runtime follow the same path even after `seq_cp`. Backend support is checked using the recurrent cache's device. Rejected indexed inputs use the existing gather-then-plain graph.

CUDA accepts indexed multi-sequence input, but disables snapshot cache fusion for it. Each sequence can read a row another sequence would overwrite; separate output and cache-copy nodes uphold that ordering. The first real node after GDN must be its snapshot copy before CUDA can fuse it. A nonempty tail gather or relocation prevents that fusion automatically.

## Validation obligations

CPU equivalence cases compare indexed GDN with `GET_ROWS` plus plain GDN. They check the residual against zero even when both backends are CPU, and exclude unwritten snapshot slots. The row-0 mutant must fail. CUDA must run those cases and sanitizer items before this getter form has GPU evidence.

The shared-sequence reserve test enables rollback snapshots and decodes one copied child while the other is idle, forcing extra-row preservation. The separate CPU server test uses NO_REALLOC, multiple completion slots and MTP; its trace must show `seq_cp` engagement. The base's known split-replay failure remains a separate issue.

## Open items

- Run the prepared stack6 sanitizer and depth items, then test the rebased master form on GPU. Old opcode-form GPU numbers are not getter-form results.
- CUDA/HIP/MUSA and rejecting accelerator backends require platform validation. Other backends use gather-then-plain, which can retain their prior snapshot limitations.
- Maintainer acceptance of the ggml signature change and CPU/CUDA bundling is unverified. No network duplication or discussion check was performed under this task's no-network rule.
- The parent installs this staged replacement into the canonical package 67. Keep the previous files under `alt-opcode/`; remove the old root opcode patches and old root `on-63/` before copying the new primary files. There is no independently tested on-package-63 getter variant here.
