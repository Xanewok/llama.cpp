# Per-request MTP draft vocabulary extension (design)

Base: stack6 `ac7465885`. Branch `topic/draft-vocab-ext`. Default off. Build and test: `codex/bw-vocab-ext.sh`.

## Problem (from `work/draft-vocab-cov`)

- The reduced draft head (`--spec-draft-vocab`, 25879 rows) misses about 2 % of the emitted tokens. These tokens end 7–9 % of
  the rounds (M).
- 66–75 % of the missed tokens already occur in the prompt of the same request. A prompt has about 134–396 distinct ids
  outside the reduced vocabulary (M).
- An oracle that adds exactly the prompt's out-of-vocabulary (OOV) ids gains +2.8 … +3.9 % tokens/round (D).
- The full head costs +1.2 ms per draft step at np1 and +6.2 ms/round at np2 (M/D). Full head = 248320 rows.

## Decision: (c) a fixed extension pool inside the reduced head

The model allocates the reduced head with `N_pool` extra rows (`--spec-draft-vocab-ext N_pool`). These rows are part of
`draft_output.weight`, and their ids are part of the draft id map. At prompt end, each sequence writes its ranked prompt
OOV ids into its quota of the pool (`N_pool / n_seq_max`).

Every pool row always holds a distinct valid token id. When no prompt id fills a row, the row holds a filler id: first the
id the row held before, then the lowest unused ids. The pool therefore needs no padding rows and no mask.

The draft graph, the logits shape, the backend top-k, the TP candidate gather and the CPU fallback do not change: they
already handle a reduced head of any size with any id map. Only the bytes of `N_pool` head rows and of the id map change,
between two graph executions.

Rejected candidates:

- **(a) A per-sequence extra head concatenated before top-k.** Under `-sm tensor` the reduced logits are split along
  axis 0. The meta backend cannot concatenate along the split axis: `handle_concat` returns UNKNOWN and aborts. The TP
  `TOP_K` rule also accepts only a simple, single-segment split. A per-column head (one weight plane per sequence) needs
  `MUL_MAT_ID` and a per-column id table. This means new graph inputs, a new CPU fetch path and new sampling code.
- **(b) A separate shared pool tensor.** It has the same concat problem as (a). To keep this tensor outside the head and
  still work under TP, the pool must be mirrored, with a second candidate set merged after the TP gather. This changes
  `build_sampling`, the logits fetch, `expand_draft_logits` and the qwen35 MTP graph. The chosen design (c) is (b)
  without these changes: the pool is shared like (b), and it is stored inside the existing head.

## Where the rows come from

- The source is the MTP LM head as loaded: `shared_head_head`, or `output.weight`. The pool copies the quantized bytes of
  row `id` (`ggml_backend_tensor_get` at `id * row_bytes`) into pool row `p` (`ggml_backend_tensor_set` at
  `(n_base + p) * row_bytes`). The GGUF file is not read again, and nothing is dequantized. The bytes are the same as the
  bytes that the load-time reduced-head build copies from the GGUF.
- Under `-sm tensor`, both tensors are split along axis 1 by the meta backend. The meta buffer accepted only whole-chunk
  reads and writes for split tensors. A single head row asserted (`offset % nb[2] == 0`).
- Commit 1 adds range reads and writes for split tensors whose only non-trivial outer extent is the split axis, for
  example a 2-D weight split by rows. On device `j`, the logical range `[off, off+size)` maps onto one contiguous range
  per device, because the logical bytes are the concatenation of the device chunks. The range must be aligned to
  `nb[axis]` (whole rows, or whole blocks on axis 0). This is a general meta-backend capability. Whole-chunk transfers are
  unchanged.
- This is not a gather on the device. There is no device-side op for it: `GET_ROWS` on an axis-1-split source with
  mirrored ids aborts in the meta backend, and `SET_ROWS` asserts that the destination is not split along axis 1. A host
  round trip of at most `quota` rows per request costs a few MB, once per request (see Cost).
- A repacked CPU head (`CPU_REPACK`, which has no `get_tensor`) cannot be read back. Loading with the extension enabled
  refuses this case. Use `--no-repack`. The meta CPU buffers in a CPU `-sm tensor` run are plain buffers, and plain buffers
  work.

## TP split of the extra rows

- The pool is the tail of `draft_output.weight`. The axis-1 split therefore puts the whole pool on the last device:
  26391 rows split 50/50 puts the boundary at about row 13196.
- The cost is an imbalance of `N_pool` rows on one device: about 5.5 µs per draft step at `N_pool = 512` (D, see Cost).
  This is not worth a placement rule.
- The upstream form keeps the tail. If a measurement shows the imbalance, a local variant can place the pool in the middle
  of the head, so that the split boundary bisects the pool.

## Graph reuse and shape stability

- `N_pool` is fixed at model load. The head tensor, the id tensor (`draft_output_vocab_ids`) and their addresses never
  change. The graph topology and all shapes are the same as with a plain reduced head of `n_base + N_pool` ids.
  Therefore graph reuse and CUDA-graph capture are unaffected.
- The update runs between graph executions. It synchronizes the context, then flushes deferred logits and expands
  pending draft rows with the old id map. Only after that does it write rows and ids.
- A draft round that is still in flight keeps its q: the probabilities are copied to the host (`dp.result_q`) when the
  draft is made.
- No mask is needed because no pool row is ever empty or a duplicate. Duplicate ids would double-count one token in the
  top-k candidates, and the rejection step would read the wrong q.

## Ownership: one MTP context, map in the model

- The row bytes live in the shared `model.draft_output`, so the id map lives next to them in `model.draft_vocab_ids`.
  `llama_model::draft_vocab_ext_place` writes a row's bytes and its id in one call. A context created later (for example
  a re-created `ctx_dft`) reads a map that matches the rows, whatever an earlier context placed.
- At most one MTP context per model holds the extension rows. The constructor acquires them
  (`llama_model::draft_vocab_ext_acquire`, under a mutex) and fails if another context holds them; destruction releases
  them, also when the constructor throws. Target contexts and models without extension rows are not affected.
- Why not a model-level map shared by several MTP contexts: each context keeps its own device copy of the ids
  (`sampling.vocab_ids`) and may have a graph in flight that reads the rows. Sharing would need cross-context
  synchronization and invalidation of those copies. The server uses one `ctx_dft` per model, so the single holder costs
  nothing there, and the rule is easy to state upstream.
- The per-sequence lists stay in the context, because sequences belong to it. A new holder starts with empty lists; rows
  keep their ids until one of its sequences needs them.

## np2 (two slots, different prompts)

- There is one pool for the context, with a quota of `N_pool / n_seq_max` per sequence. The layout is rebuilt on every
  update from all sequences' current lists, in sequence order:
  1. Ids that are in the reduced base, repeated, or already placed are skipped.
  2. Each list is cut to its quota.
  3. Rows that keep their id are not rewritten. Only new ids are copied.
- Every sequence drafts over the union of all prompts' ids. An id from the other prompt is one more real head row, so the
  head stays a subset of the full head. The draft can only move toward the full-head draft. This is allowed, and it
  costs nothing extra.
- Slot B's update can change the pool while slot A has a round in flight. This is exact because A's q is already on the
  host. A's later drafts use the new pool, and those drafts carry their own q.
- Draft content depends on what the other slot did before (filler ids are kept). Under probabilistic acceptance, np2
  token identity across runs already depends on batch timing (`identity_off_reason` in the np2 items). At temperature 0,
  the output does not depend on the drafts in exact arithmetic (batch shapes still change kernel numerics).

## When the set is built

- `common_speculative_begin(seq, prompt)` runs once per request, after the prompt is evaluated and before the first
  draft. The server passes the full prompt, including a cached prefix.
- The draft-MTP implementation counts each prompt token id and drops the ids in the reduced base. It ranks the rest by
  count (descending), then by last position (latest first), then by id. Then it calls
  `llama_set_draft_vocab_ext(ctx_dft, seq, ids, n)`. The library keeps the first `quota` ids that are not in the base.
- `N_pool` is the cap: 512 per slot at np1 covers the measured 134–396 OOV ids of a prompt (M) with margin. Ranking by
  frequency matters only when a prompt has more OOV ids than its quota.
- A repeated or cached request with the same prompt writes no rows.

## Exactness

- Speculative decoding stays exact for any draft distribution q. The conditions are that every proposed token was
  sampled from q, and that the verifier uses that same q.
  - Greedy drafting: the verifier samples from the target and matches. The draft only decides how many target samples
    are compared, so the output is the target's output.
  - Probabilistic drafting (`--spec-draft-sampling probabilistic`): the draft sampler takes top-k(10), then temp, then
    dist, over the head's logits. The candidates and probabilities that the sampler actually used (`cur_p`) are stored as
    `dp.result_q`. Rejection (`common_sampler_sample_and_accept_n_rejection`) accepts with min(1, p/q) and resamples from
    norm(max(0, p−q)) with that stored q. The extension changes which rows produce logits, so it changes q. q is still
    exactly the distribution that was sampled.
- Under TP, the per-shard top-k plus the gather keeps a superset of the global top-k. This is unchanged, because the pool
  rows are part of the shard slices.
- Things that would break exactness, and why they cannot happen here:
  - a duplicate id in the head: excluded by construction, and checked in the test;
  - an id map that disagrees with the row bytes: the model holds both and one call writes both, after synchronization
    and before the next graph; only the one holding context writes them, and a later context reads the model's map. The
    wrong-row mutant and the test's re-created context check this;
  - changing the head between draft and verify of the same round: q is already stored on the host.
- `--spec-draft-vocab-ext 0` (the default) allocates nothing, acquires nothing and calls nothing. The code paths are then
  those of stack6: `n_base == size`, and the draft-MTP implementation keeps no head bitmap and does no per-round
  bookkeeping.

## Cost vs gain (D)

- Per draft step: `N_pool` extra head rows. From the full-head measurement: 1.2 ms / 222441 extra rows = 5.4 ns per row
  per step at np1 (D, linear).
  - np1, N_pool = 512: +2.8 µs/step, +14 µs/round at k=5.
  - np2: 6.2 ms/round / 222441 = 28 ns per row per round. N_pool = 1024 gives +29 µs/round.
  - The pending cov99 A/B (+2831 rows) decides between this linear model and the conservative one (0.66 ms per 2831 rows,
    which gives +0.12 ms/round at 512 rows).
- Per request: at most `quota` rows, each one D2H plus one H2D of `row_bytes` (about 4–5.5 KB). 512 rows is about 1000
  small synchronous copies, about 10–20 ms (G). This is negligible against the prefill of a real prompt (seconds).
  Repeated prompts copy nothing.
- Memory: `N_pool × row_bytes` (about 2–3 MB for 512 rows). It is split over the devices.
- Gain: the oracle gives +2.8 … +3.9 % tokens/round (D). The ranked prompt set is that oracle's id set whenever the prompt
  has ≤ quota OOV ids. At about 45 ms/round (G, from the realprompt-depth notes), the per-step cost is under 0.1 % (D,
  linear) or under 0.3 % (D, conservative). The expected net change is +2.5 … +3.9 % tok/s.

## Interfaces (upstream form)

- `llama_model_params.n_draft_vocab_ext`: the number of pool rows. It requires `n_draft_vocab > 0`, and the pool must fit
  in the vocabulary.
- `uint32_t llama_draft_vocab_ext_capacity(const llama_context *)`: `N_pool` for an MTP context with a pool, else 0.
- `int32_t llama_set_draft_vocab_ext(llama_context *, llama_seq_id, const llama_token *, size_t)`: replaces the
  sequence's ids. It returns how many of them the head holds, or -1 on error. The ids must be in priority order.
- A second MTP context on a model with extension rows fails to construct while the first exists.
- common: `--spec-draft-vocab-ext N` (env `LLAMA_ARG_SPEC_DRAFT_VOCAB_EXT`). It is ignored with a warning without
  `--spec-draft-vocab` or with a draft model. The draft-MTP `begin()` ranks the prompt and calls
  `llama_set_draft_vocab_ext`. With N > 0, process-wide counters (`common_speculative_get_vocab_ext_stats()`) and one
  INFO line per request report engagement: `draft vocab ext: seq S: P prompt OOV ids, H in the draft head` (the library
  logs the rows copied).
- Local only (separate commit): `LLAMA_DRAFT_VOCAB_EXT_MUTANT=row` copies row `id+1` instead of `id`. It is the
  wrong-gather negative control for the tests.

## Tests (CPU, Qwen3.5-4B UD-Q4_K_XL, which has one MTP layer; `-c 4096`)

- `codex/test-draft-vocab-ext.cpp` (llama API, host and backend top-k, plain and Meta(CPU,CPU)):
  - the pool rows' logits equal the full head's logits for the same ids;
  - ids outside the head are −inf;
  - the backend top-k matches the host top-k over base + pool;
  - an update replaces ids without duplicates, and two sequences share the pool within their quotas;
  - with ext off, every row's head support is exactly the base ids, so no out-of-head id can be drafted;
  - a context created after the placing one was freed reads the placed ids, and its rows match the full head;
  - a second MTP context on the extension model is refused while one exists; a target context is not;
  - the mutant must fail the logit check.
- `codex/bw-ext-driver.cpp` (server_context, as in host-trims-v2), plain and Meta, greedy and probabilistic:
  - **off identity**: ext off gives the same tokens and draft counters as the stack6 base source (`git archive` of
    `ac7465885`), and two runs of the base repeat;
  - **engagement**: with ext on, the per-request ids placed are > 0;
  - **OOV acceptance**: with a reduced head of 1/17 of the vocabulary, the accepted draft tokens outside the base vocab
    are > 0 with ext on (copy task, greedy). With ext off the counters stay 0 because N = 0 does no bookkeeping; that
    no out-of-head id can be drafted then is the unit test's support check;
  - not gated: greedy tokens with ext on vs off. Speculation is exact in exact arithmetic, but the arms verify
    different batch shapes and CPU kernels change summation order with the batch size, so near-ties may flip. The
    common token prefix is reported (M).

## GPU A/B (items generated, not submitted)

- Arms: stack6 (frozen) vs `frozen/draft-vocab-ext` with `--spec-draft-vocab-ext 512`. Setup: J5q TP2 np1 k=5, ub512,
  real-prompt depth harness (`wt/harness-real-prompt`).
- Identity is within-arm (the harness allows "off" only for np > 1). The arms draft differently, so their probabilistic
  token streams differ. Within the ext arm the pool depends on request history, but every boot loads the same filler rows
  and runs the same depths in the same order, so each boot must repeat the arm's tokens and counters. A failure there
  means hidden non-determinism in the pool update, not noise.
- Judge on tokens/round, acceptance and ms/token. The canary proves engagement with the INFO line.
- Corpus: `tokens64k` (6 sessions) is from tasks that are in the analysis corpus, so it is in-sample.
  - The item generator also renders a `tokens8k` item (8k/32k). It is held out only if the generator's n-gram overlap
    check against `tokens64k` passes. See `codex/gen_items_vocab_ext.py`.
