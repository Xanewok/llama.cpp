# Teacher-forced MTP acceptance

`llama-spec-tf` records target probabilities and MTP draft probabilities on fixed token streams. It does not generate text or measure GPU speed.

Pass normal server model/context/sampling arguments, plus:

- `--tf-stage A --tf-streams streams.json --tf-out target`: record target hidden rows and probabilities once.
- `--tf-stage B --tf-target target --tf-out arm`: reuse those hidden rows for another draft model or `--spec-draft-vocab`.
- `--tf-stage AB --tf-streams streams.json --tf-out output`: run both stages in one process. Each repeated `--tf-control-vocab ids.json` adds a real sliced-head draft pass in `control-N/`.

A stream file is a JSON list of objects with `id` (optional string), `prompt` (nonempty token-id list) and `continuation` (token-id list). Token ids refer to the model vocabulary, including any prompt special tokens. Use an explicit context large enough for the complete stream and `-np 1`. Inputs must use the same target model and sampling settings as the recorded stream. For the current estimator use positive temperature and a top-k bounded target chain; stochastic stateful samplers, grammars and reasoning-budget forcing are rejected.

For example, with an MTP-enabled GGUF:

```sh
llama-spec-tf -m model.gguf -c 4096 -np 1 -t 8 -tb 8 -fa on -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --temp 1 --top-k 20 --top-p 0.95 --min-p 0.05 --ignore-eos \
  --backend-sampling --spec-draft-backend-sampling \
  --tf-stage A --tf-streams streams.json --tf-out target
llama-spec-tf -m model.gguf -c 4096 -np 1 -t 8 -tb 8 -fa on \
  --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-type-k f16 --spec-draft-type-v f16 --temp 1 \
  --tf-stage B --tf-target target --tf-out base
python3 codex/spec_tf.py --target target --arm base --output estimate
```

Each output directory has a small `header.json` written only after successful completion. Each numbered stream directory contains little-endian binary arrays, with dimensions in the header:

| Stage | File | Type and shape |
|---|---|---|
| A | `tokens.bin` | I32 `[n_tokens]` |
| A | `hidden.bin` | F32 `[n_tokens, n_embd]` |
| A | `p_ids.bin`, `p_probs.bin` | I32/F32 `[n_tokens, p_width]` |
| A | `p_raw_ids.bin`, `p_raw_logits.bin` | I32/F32 `[n_tokens, 24]` |
| A | `p_logsumexp.bin` | F32 `[n_tokens]` |
| B | `starts.bin` | I32 `[n_continuation - 1]` |
| B | `q_ids.bin`, `q_logits.bin` | I32/F32 `[n_continuation - 1, k, 128]` |
| B | `q_sample_ids.bin`, `q_probs.bin` | I32/F32 `[n_continuation - 1, k, 10]` |

A target row at position `p` predicts token `p+1`. A draft start `t` feeds token `t` and target hidden row `t-1`; later steps feed the true stream tokens and the draft's own hidden rows. The saved draft top-128 supports offline vocabulary variants only when their top ten unsuppressed ids are covered. `q_sample_ids` and `q_probs` preserve the actual `common_sampler` row for real sliced-head runs. Padding uses id `-1` and probability zero. The estimator excludes starts without a complete draft horizon; their partial rows remain available for checking final server rounds.

`--tf-hidden-offset -2` and `--tf-permute` are negative controls. They intentionally break the draft conditioning. Independent MTP contexts and chained MTP heads use the existing draft driver conventions; assistants that require a live target context are unsupported. Quantized inference can depend on batch shape. A row comparison against `LLAMA_SPEC_TRACE` must pass before treating the result as a paired estimate of a served configuration.

Use the served backend-sampling flags: a CPU backend can run the full target chain, while a tensor-split backend can offload only a prefix. `--tf-chunk N` sets the continuation decode width for batch-shape diagnostics without changing `-ub` or the prompt pass. Its default is the physical batch width.

Stage A mirrors the server's final checkpoint prompt splits and records the physical prompt chunks for Stage B. For an uncached sequence of requests from one server boot, use `--tf-carry-streams` in Stage B (or AB), with streams in request order. The current server keeps `pending_h` across requests, so the first prompt row uses the previous request's final decoded target hidden. The default starts each stream with zero carry, as in a fresh server boot. Matching the served continuation width (`--tf-chunk 6` for k=5) avoids a CPU quantized batch-shape difference.
