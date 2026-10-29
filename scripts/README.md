# Evaluation runners

These Python 3.11+ scripts use only the standard library. Result and prediction files are append-only: use new paths for a fresh experiment.

The directory map is by output: `infra/` contains shell and systemd support, `instruments/` produces rows, `analysis/` produces tables, and `corpora/` selects tasks and proves leakage boundaries. Run Python tools as package modules from the repository root, for example `PYTHONPATH=. python3 -m scripts.instruments.run_polyglot`.

## `emit.py`

`emit.py` exposes `RunContext`, `emit(ctx, task_id, run_index, **measures)`, and `scrape_metrics(base_url)`. Every call opens the output in append mode, takes an exclusive `flock`, writes one fixed-shape JSON row, flushes and `fsync`s it, then closes the file. Unknown fields are rejected; unspecified fields are JSON `null`.

The metrics helper accepts either a server root or an OpenAI base URL such as `http://127.0.0.1:8000/v1`. It tolerates an unavailable `/metrics` endpoint and returns any available `prefill_tps` and `decode_tps` gauges.

```bash
PYTHONPATH=. python3 -m scripts.instruments.emit --out results/smoke.jsonl --task-id smoke --run-index 0 --lane qwen --quant Q4_K_M
```

## `run_polyglot.py`

The Polyglot runner copies each source tree to a temporary directory, invokes Codex, runs the exercise test command, and allows exactly one Codex repair and retest after an initial failure. A task exception or timeout emits a failed row and does not stop later tasks. Rows retain manifest order even with `--jobs` greater than one.

Run the mandatory harness self-test first. It overlays each known-good Exercism example onto its solution path and requires that copy to pass; it separately requires the untouched stub to fail. The report includes every task, including broken ones, and the command exits non-zero if any task is broken.

```bash
PYTHONPATH=. python3 -m scripts.corpora.selftest_polyglot \
  --manifest subsets/polyglot.json \
  --out results/selftest/polyglot-selftest.json
```

Pass the successful report to every benchmark run:

Local lanes (`qwen`, `quant`) require `--env-json`, a per-leg capture from `scripts/instruments/gpuenv.py`; `--dry-run` and hosted lanes (`frontier`, `control`) do not. The `emit.py` command-line smoke test above takes no `--env-json`.

`--lane qwen` (alias `quant`) selects `-p qwen38`; `--lane frontier` (alias `control`) selects `-m gpt-5.6-luna`. Both use the same headless invocation path and pass `--reasoning-effort` to `model_reasoning_effort`.

```bash
PYTHONPATH=. python3 -m scripts.instruments.run_polyglot \
  --manifest subsets/polyglot.json \
  --selftest-report results/selftest/polyglot-selftest.json \
  --lane qwen --quant Q4_K_M --run-index 0 \
  --env-json env.json \
  --out results/polyglot.jsonl --jobs 2
```

The manifest is one JSON array. `source_dir` is absolute or relative to the manifest. `test_cmd` is either a command string parsed with `shlex` (without a shell) or an exact argument array; pipes and redirections therefore require an explicit shell argument array. Optional `example_path` and `solution_path` values are relative to `source_dir`; each may be one string or a non-empty array with matching cardinality. When omitted, the self-test derives them from `.meta/config.json` at `files.example` and `files.solution`. A manifest generator should populate the optional fields from those same metadata keys.

```json
[
  {
    "task_id": "python-two-fer",
    "language": "python",
    "source_dir": "exercises/python/two-fer",
    "test_cmd": "python -m unittest -q"
  },
  {
    "task_id": "rust-gigasecond",
    "language": "rust",
    "source_dir": "exercises/rust/gigasecond",
    "test_cmd": ["cargo", "test", "--", "--include-ignored"],
    "example_path": ".meta/example.rs",
    "solution_path": "src/lib.rs"
  }
]
```

Rust commands are forced to include `--include-ignored`. For JavaScript, every `xtest(` in the test files listed by `.meta/config.json` is rewritten to `test(` inside each scratch copy before execution. These rules apply identically to self-testing and benchmark scoring.

The report is a JSON object with `schema_version`, `manifest`, `harness_verified`, and a `tasks` array. Each task record contains `task_id`, boolean `reference_passes`, boolean `stub_fails`, boolean `broken`, and captured details under `reference` and `stub`. `run_polyglot.py` refuses a missing, malformed, incomplete, or broken report. `--skip-selftest` exists only for development, prints a prominent warning, and emits `harness_verified: false`; verified runs emit `true`.

Use `--dry-run` to copy every task and execute its tests without invoking Codex. This validates paths, commands, timeouts, ordering, and result emission without a GPU, but it still requires a successful self-test report unless `--skip-selftest` is explicitly supplied.

## `run_swebench.py`

The SWE-bench wrapper maintains bare mirrors under `--repo-cache`, makes a disposable shared clone per instance, checks out and cleans the requested base, invokes Codex, and appends the resulting Git diff in SWE-bench prediction format. It only generates predictions; it does **not** run SWE-bench evaluation containers.

```bash
PYTHONPATH=. python3 -m scripts.instruments.run_swebench \
  --instances subsets/swebench.jsonl \
  --lane qwen --quant Q4_K_M --run-index 0 \
  --env-json env.json \
  --out results/swebench.jsonl \
  --predictions results/swebench-predictions.jsonl \
  --repo-cache .repo-cache
```

The instances file is JSONL with exactly one object per line. `repo` may be a Git URL, an `owner/name` GitHub shorthand, or a local path; an existing relative local path is resolved relative to the instances file.

```json
{"instance_id":"django__django-12345","repo":"https://github.com/django/django.git","base_commit":"0123456789abcdef0123456789abcdef01234567","problem_statement":"Fix the documented regression and add coverage."}
```

Prediction rows have this exact shape:

```json
{"instance_id":"django__django-12345","model_name_or_path":"qwen3.8-27b","model_patch":"diff --git ..."}
```

Use `--dry-run` to exercise cache setup, clone, checkout, clean, diff, prediction emission, and result emission without invoking Codex. An unchanged checkout produces a valid empty prediction plus a failed result row explaining the empty diff.

Run all offline tests with:

```bash
python -m unittest discover tests
```

## `build_kld_corpora.py`

This builder fetches `wiki.test.raw` plus ten Git repositories at the exact revisions in `corpora/sources.json`. It applies the documented extension, path, `.gitattributes`, size, line-length, UTF-8, test-cap, exact-dedup, and 128-permutation MinHash rules. Surviving code files are byte-path sorted inside each repository and interleaved round-robin in manifest repository order. Files are joined without headers, and manifest byte offsets are half-open (`byte_start <= position < byte_end`).

The timestamp is always caller-supplied. A normal build writes tracked corpus bytes, committed JSONL attribution manifests, and hashes in `BUILD.json`:

```bash
PYTHONPATH=. python3 -m scripts.corpora.build_kld_corpora \
  --timestamp 2026-08-20T00:00:00Z \
  --cache-dir .cache/kld-corpora
```

Rebuild from the pinned sources and compare the corpus metadata and every attribution manifest without overwriting anything:

```bash
PYTHONPATH=. python3 -m scripts.corpora.build_kld_corpora --verify --cache-dir .cache/kld-corpora
```

The source cache makes the verification run network-free after the initial fetch. The corpus text files `corpora/*.txt` are tracked, not ignored; `BUILD.json` holds their hashes. Each committed manifest row is `{"repo", "sha", "path", "sha256", "byte_start", "byte_end"}`.

## `make_subsets.py`

This generator materializes `subsets/polyglot.json` (40 runner-ready tasks), `subsets/diffxyz.jsonl` (50 source rows carrying separate `udiff` and `search-replace` targets), `subsets/swebench.jsonl` (10 runner-ready instances with Docker image names), and `subsets/BUILD.json`. The seed and all source revisions and canonical dataset-export hashes are frozen in `subsets/sources.json`.

```bash
PYTHONPATH=. python3 -m scripts.corpora.make_subsets --cache-dir .cache/subsets
```

Diff-XYZ and SWE-bench Multilingual are Parquet-only upstream. To stay standard-library-only, the first run downloads their Hugging Face JSON row views, rejects truncated cells, canonicalizes them to sorted-key JSONL, and requires the pinned SHA-256 before selection. Later runs use the cached JSONL. For a fully offline rebuild, provide canonical exports explicitly:

```bash
PYTHONPATH=. python3 -m scripts.corpora.make_subsets \
  --diffxyz-jsonl /data/diffxyz-pinned.jsonl \
  --swebench-jsonl /data/swebench-multilingual-pinned.jsonl
```

`polyglot.json` is the JSON array consumed directly by `run_polyglot.py`; its `source_dir` values point into the default cache. `swebench.jsonl` is consumed directly by `run_swebench.py`. A Diff-XYZ scoring harness should run each of its 50 rows twice, asking for `udiff` and `search-replace` independently rather than combining their scores.

## Result file hygiene

`emit()` takes an `flock` around every append. Anything else that touches a live
result JSONL must take the same lock, or normalise missing keys at read time
instead of rewriting the file -- a schema change applied by rewriting the append
target races the runner and can lose a completed task.
