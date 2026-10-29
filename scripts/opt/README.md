# opt: results DB, feasibility fit, wall_est

Stdlib only (Python >= 3.11 for `tomllib`). CPU only. Entry point: `python -m scripts.opt <command>`.

```
python -m scripts.opt [--db DIR] ingest PATH... [--dry-run] [--strict] [--show N]
python -m scripts.opt [--db DIR] status
python -m scripts.opt [--db DIR] fit                    # VRAM fit + HQ pool rule, residuals
python -m scripts.opt [--db DIR] profile                # pool call tables into profile:* workloads (writes the DB)
python -m scripts.opt [--db DIR] calibrate              # wall_est error against recorded campaign walls
python -m scripts.opt [--db DIR] report GOAL.toml [--all] [--limit N]
```

`--db` defaults to `results/db` (override with `$OPT_DB`). Tests: `python -m pytest tests/opt -q`
(fixtures are trimmed real files under `tests/opt/fixtures/`; `PROVENANCE.txt` there lists their origin).

## Tables (`results/db/*.jsonl`, one JSON object per line, sorted by id, written atomically)

| file | key | content |
|---|---|---|
| `measurements.jsonl` | `id` | `ts, source, item_id, attempt, config_id, workload_id, host{}, metric, dims{}, value, unit, n, spread, pair_group, status, reason?, provenance{out_dir, summary_path, summary_sha256, build_sha, model_sha256, harness_sha, parser, parser_version, also_at?}` |
| `configs.jsonl` | `config_id` (= `id`) | `engine (llama\|hq\|weights), stack_name, build_sha, model_label, model_path, model_sha256, canon{}, canonical_toml` plus derived `np, ctx, ctx_per_slot, kv_k, kv_v, k, p_min, ub, gpu_util` |
| `workloads.jsonl` | `id` | `task-set` rows, `campaign-calls` / `two-agent-calls` call tables, and `profile:*` pooled profiles |
| `levers.jsonl` | `lever_id` | written by another tool, not by this package; `Store` loads it and never rewrites it |

* **Row ids are content-derived** (source, source digest, config, workload, metric, dims, key), so re-ingest is a no-op
  (`added 0, updated 0`, byte-identical files). The same file seen at several paths (`results/imports/...` archive
  copies) merges into one row; the repo-relative path is canonical and the others go to `provenance.also_at`.
* **`config_id`** = first 16 hex of sha256 of `canonical_toml` of `canon = {engine, model{label, sha256|path}, flags{}, env{}, stack, build{sha, binary_sha256, vllm_version}, chat_template_sha256}`.
  Ports, hosts, scratch paths, device UUIDs and aliases are excluded; `-m` is replaced by the model sha256 when the
  launch record has one. Flags are stored without dashes; JSON-valued flags (vLLM `speculative-config`) are nested.
* **Model identity** (`configs.model_identity`): `model_id = <family>-<version>:<recipe>`, e.g. `swift-1.5:IQ4_XS-ssmout6`,
  `swift-1.0:W4A16-syv-fast`; `Swift-Qwen3.8-27B-...` without a version is 1.0; anything else is `other:<stem>`. Same quant label with a
  different version/recipe/sha is a different model. **Candidate key** = (engine, stack, model_id, model sha, KV, k, p_min, ub, np,
  ctx/slot, gpu_util, `variant_id`); `variant` = every flag outside the keyed set plus `GGML_/VLLM_/...` env, so A/B env arms are never pooled.
  A sha-less member merges into the unique sha group of its key.
* **`status`**: `measured` (read from a source number), `derived` (computed from measured numbers, including
  every `*_D` field of the producers and all model outputs), `ledger-prior` (copied from the notes, never from a run),
  `invalid` (the run exists but is not usable: stopped, canary, wall-sum mismatch, failed verdict; `reason` says why).
  Invalid rows are kept so a bad run is visible, and are excluded from fits, profiles and reports.
* **Reference is part of the KLD metric**: `kld@REF6` (Q6_K, f16 KV), `kld@REF8` (Q8_0, q8_0 KV), `kld@REFBF` (BF16, f16 KV).
  A table whose reference cannot be read from its own header is skipped, never defaulted. Harness-local references
  (`ref_local`, `ref_remote`, `ref_q6k`, ...) are kept literal and are not comparable with the three above.
  A `results.txt` with a sibling kld `results.json` is a rendered view and is skipped; a `results.txt` alone is parsed.
* **KLD join** is exact on `model_id` and exact on KV (`kld_for`); there is no fallback to another KV. A KLD at another KV is listed as
  context (`other KV: ...`) and the constraint stays unverified. Arm labels map to (recipe, KV) through the explicit `ARMS` table in
  `ingest/kld.py` (each entry cites its evidence); an arm outside the table gets `model_id None` and never joins. Control arms
  (`CONTROL_ARMS`, the J4 base row of a delta table) and variant arms (`FA_F32_ACC` on) are stored but never feed a verdict.
* **`kld@ledger-q6k-8k`** rows (`ingest/ledger.py`, status `ledger-prior`) are the PLAN.md HQ KLD numbers at the 8k Q6_K scale, re-verified
  by regex at each ingest. Own reference key: shown as context in the report, never compared with `kld@REFBF`, never a PASS.

## Sources

| source | unit ingested | notes |
|---|---|---|
| `campaign` | dir with `SUMMARY.json` + `manifest.json` + `job-*.json` | name = `dir@sha8` (same-named dirs with different content never collide); a job without `probe` is a not-run task (`task_solved 0`, `ran false`, reason kept, campaign `invalid`); a passed job without a probe is rejected; sha256 sidecars verified; `wall_seconds` must equal the sum of task walls (else rows are `invalid`); call table from `attempts/*/model-calls.jsonl` when present |
| `weekend` | `summary.json` list | campaign dir parsed when present, otherwise config rebuilt from the row (flagged) |
| `graph-reuse-ab` | `ab-summary.json` + per-boot `command.json` | `_M` fields measured, `_D` derived |
| `hq-conc` | boot dir (`fit.json`, `serve.log`, `single-*`, `conc-*`, `prefix.json`) | failed prefix check = `prefix_kept 0`; canary `prefix.json` = `invalid` |
| `two-agent` | run dir, or its `measure/`, whose `SUMMARY.json` has `agents` | per-agent call tables kept; `memory.csv` -> `gpu_idle_used_mib` (a snapshot, not a min-free); runaway calls -> `runaway_rate`; canary-only dirs rejected |
| `queue-np` | `raw/NN-arm-rR.json` + `.launch.json` + `.smi.csv` | `round_ms = (n/server_tg)*1000/(n - accepted)` per stream, server decode clock, streams without `server_tg` skipped (matches the producers' summary.md to 0.01 ms); min-free from the smi csv |
| `kld` | `results*.txt` / `results*.json` | rows are `{kind: weights, model_id, kv}` configs joined by exact model identity and KV; row id keyed on the file's content sha, not the directory name |
| `ledger` | `PLAN.md` | seven HQ KLD numbers, `ledger-prior`; a PLAN.md without them is skipped |

A corrupt or partial source raises `Reject` and is listed; a recognised source that deliberately yields nothing raises
`Skip`. Neither writes rows. `ingest --strict` exits 1 on any reject.

## Models

* **VRAM fit** (`fit.py`): `free(gpu) = a[group,gpu] - b*ctx - c*k`, group = (engine, quant, kv, build family, ub, tensor split).
  Median over boots first. `c` (MiB per draft token) is the median within-cell slope over k. `b` is fitted per group only
  when its contexts differ by >= 20%; otherwise borrowed from groups of the same KV, otherwise scaled by KV bytes.
  `ctx` and `np` are confounded (about 100k tokens per slot), so `b` is an effective slope over total context. `opt fit` prints in-sample and
  leave-one-boot-cell-out residuals; a point is stale if |resid| > max(5% of free, 100 MiB). Band: predicted free < 200 MiB reject,
  200..500 probe needed. A *predicted* value classifies with its leave-one-cell-out band (p90 |resid|, floor 100 MiB; 528 MiB on the real DB): reject only if even `free+band` is below 200, `fits` only if `free-band` clears 500. HQ candidates also need the predicted/measured KV pool per slot (`pool/np`) >= the goal's ctx_per_slot, not only `max-model-len`.
* **HQ pool**: `pool = pool0[ckpt] - per_k*(k-5) + 860000*(util-0.86)`. `per_k` is fitted from boots at two k; the util slope is a
  ledger-prior (PLAN.md:642) because every recorded boot is at util 0.86. Prefix kept at depth <= 0.302 pool, lost at >= 0.315 pool
  (observed brackets at pool 199033). The rule is IN-SAMPLE (thresholds set from the same probes); depth is rounded up to whole `prefix-match-unit` blocks.
* **wall_est** (`workload.py`): over a pooled call table, per call `fixed + new/prefill_tps(depth) + out/decode_tps(depth) + gap`, plus a per-task overhead,
  averaged over the pooled campaigns. `fixed` (HQ) and the task overhead are fitted from the HQ call tables; the llama fixed cost 0.54 s is a ledger-prior (no llama call tables exist).
  Speeds: `campaign-agg` (measured aggregate of a campaign of that config), `two-agent`, or `F1` (steady-round depth curves, piecewise linear, flat beyond the recorded depths).
  Llama candidates use the HQ call shape with token volume re-based to the median llama campaign.
  Two agents: slower of the two streams; `rho` = 0 is fitted on n=2 runs (reproduced within 1% in-sample, `rho` = 1 over-predicts by 6-13%), not validated.
  `opt calibrate` tiers: A and B0 are in-sample structure checks; **B is held out** (other configs' calls and constants, speeds from replicate
  runs of the same config, never the run's own speeds; needs a replicate); C0/C/D are llama-side (D = F1 depth curves, a true prediction); DvC has no
  real wall. Each tier prints median/max |error| and a Spearman rank correlation (average ranks). Tie-band width: hq = tier B median, llama = tier C
  median, F1 = tier D median, two-agent = the rho 0..1 span (the in-sample error would be ~1%). Rows within the width of a band's best estimate
  share a band and are not ordered by wall_est.

## goal.toml

```toml
[goal]
name = "fidelity-first one agent"
workload = "codex-8task-1agent"          # or two-agent-8task
primary = "wall_est"                     # only scalar with a cost model; minimized
tiebreak = ["kld@REFBF", "vram_min_free_mib"]   # lexicographic; direction is fixed per metric
[constraints]
fits = { np = 1, ctx_per_slot = 98304 }  # candidate np and ctx per slot must be >= these
kld_REFBF = { max = 0.0035 }             # kld_<REF> -> kld@<REF>
vram_min_free_mib = { min = 300 }
runaway_rate = { max = 0.15 }
headless_only = false
```

Optional `model = ["swift-1.5:*"]` (fnmatch patterns on `model_id`, each containing `:`) restricts candidates; excluded ones are counted in the
report. `solve_rate = { min = 0.9 }` is evaluated from the campaign `solve_rate` rows. An unknown `workload` is a clean error (exit 2).
Unknown keys are errors. A constraint without data for a candidate is *unverified* (shown, not a pass): rows are ranked PASS first,
then candidates with unverified constraints, then (with `--all`) violators. Within a verdict, a candidate whose KLD is missing or measured at another KV ranks below every candidate with a verified KLD. Measured-basis (`campaign-agg`/`two-agent`) and F1-basis candidates are ranked in separate lists and never against each other; the measured wall sits next to every estimate. The Pareto front (wall_est x KLD) keeps candidates that do not
violate a non-KLD constraint.

## Known gaps

* Older campaign schema (2026-09-12 `mixed-pair`: `SUMMARY.json` without `tasks/solved/wall_seconds`, `job-solve-*` without `probe`) is rejected, not parsed.
* Llama campaigns have no `model-calls.jsonl`: no llama call table, no llama `runaway_rate`; llama wall_est uses the HQ call shape.
* HQ checkpoints have no REFBF KLD rows, only the `ledger-prior` context column; HQ boots have no VRAM min-free in the campaign records.
* Most KLD dumps are f16 KV while candidates run q8_0/q8_0: the exact-KV join leaves them unverified (not failed) until a q8_0-KV dump exists.
* Arms outside the `ARMS` table (A*, ref_* quant ladder, ...) never join. The J5 queue arms' `-bf16` identity is inferred from naming.
* Multi-model SUMMARY campaigns (`confirmed-fullsuite`) are still rejected. KLD ids follow file content: an edited-in-place results file leaves its old rows behind.
* Tier B needs replicates; the HQ prefix rule is in-sample; `rho` rests on n=2.
* KLD result files without a stated reference (about 29 under `results/**/imports`) are skipped; `vast-gguf-s` speed tables are not ingested.
* F1 prefill curves have one depth point per config; np2 F1 speeds are per-stream under concurrent load.
* Build identity for llama configs is `stack_name` (a path-derived family label) plus `build_sha` where the launch record has it.
