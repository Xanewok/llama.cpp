# swe-fresh calibration night (fable-bench REPORT 2.2)

Labels: M measured here, D derived, G guess. Times Europe/Warsaw (CEST). Nothing here ran on a GPU or a real model.

## What is proven (M, CPU only, stub Responses server, `pallets__click-3239`, Harbor 0.23.0, Codex 0.160.0)

| Question | Result | Evidence |
|---|---|---|
| Container -> host egress | Works: container (172.19.0.2) reached the relay bound on `172.17.0.1:<ephemeral>`; the stub stayed on 127.0.0.1 | `proof/run1-builtin`, relay rows, stub `request-*.json` |
| `--allow-agent-host` | **No-op.** Task network mode is `public` for all 860 tasks; Harbor warns "ignored because the effective network policy is public" | `proof/*/harbor.log`, `network_policy.merge_extra_allowlists` |
| Codex install in container | Works (nvm + node 22 + `npm i -g @openai/codex@0.160.0`), `agent_setup` 104-127 s per task (3 runs), not cached between tasks | job `probe.agent_setup_seconds` |
| Built-in `openai` provider + `OPENAI_BASE_URL` | Reaches the relay (rollout `model_provider: openai`), but Codex first probes a **WebSocket** (`GET /v1/responses`, `openai-beta: responses_websockets`), 7 attempts / ~7 s, 404, then falls back to HTTP POST. Model metadata for the alias is "not found" (fallback metadata) | `proof/stub-run1/get-log.jsonl`, `codex.txt` |
| `model_provider="local"` fallback via `--ak config=<json>` | **Use this.** `[model_providers.local]` with `supports_websockets=false`: no probe, no auth header, `/v1/responses` POST only, rollout `model_provider: local` | `proof/run2-local`, `proof/stub-run2/` (no `get-log.jsonl`) |
| Window reaches Codex | `model_context_window=98304` becomes 93,388 in `token_count` events (Codex reports 95%); compaction limit 88,000 set | rollout `token_count.info.model_context_window` |
| Grading | Verifier ran, `reward 0.0` for the stub, harness status `unsolved` (valid outcome, not an error) | `verifier/reward.txt`, `SUMMARY.json` |
| Snapshot vs registry | `-p data/dl/swe-rebench-leaderboard` gives the same `task_checksum` (98384bee...) as `-d`; names differ (`-d` needs `-i swe-rebench/<id>`, `-p` bare `<id>`) | run1 vs run3 |
| Dataset pin | `-d name@sha256:ebe7444e...` accepted; the registry digest is recorded per job (`probe.dataset_ref`) | `proof/run2-local` config |
| Harness outputs | SUMMARY/job schema keeps the Campaign-index fields; engine counters path (HQ-style `vllm:` counters) and per-call-timings path (llama-style) both exercised | `proof/final-a-canary`, `proof/final-b-repeat2` |
| 20 min cap | `--agent-timeout-multiplier = cap/3000` (all 860 tasks declare agent timeout 3000 s); harness gates `cap_not_enforced` if the agent wall exceeds cap x 1.1 + 60 s. **Not exercised against a real overrun** | task.toml scan; gate has a unit test |

Open and untested: (1) the egress allowlist for the agent phase (`task.toml [agent] network_mode="allowlist"`) needs Harbor's egress sidecar image (another pull; outside my one-image budget). Without it the agent has full internet in the container and could fetch the upstream fix; decide before trusting wall times (canary should grep `codex.txt` for `git clone`/`pip download`/`curl`). (2) HQ/llama real Responses SSE: only CPU-fixture-tested (two-agent recorded rows) plus the stub.

## Disk (the user must free Docker space first)

Docker root `/mnt/scratch/docker/data-root`, free space `df -B1 /mnt/scratch`:

| Moment | Free bytes |
|---|---|
| Before anything | 13,204,385,792 |
| Peak during first run (pull + build + Codex) | 8,578,236,416 (-4.63 GB) |
| After run 1, `--delete`d | 9,936,384,000 |
| After runs 2, 3, final A, final B (same image) | 9,936,384,000 .. 9,937,604,608 |

Net **-3.27 GB stays** after Harbor's `--delete`: Harbor removes the container and its built image, but BuildKit keeps the pulled base layers (`docker builder du --verbose`: records "pulled from docker.io/swerebench/sweb.eval.x86_64.pallets_1776_click-3239", reported as 0B, 11 records created by my runs, IDs in `proof/my-buildkit-ids.txt`). Later runs on the same image reuse them (env setup 2.8 s instead of 45.7 s). I did not remove them: the one prune attempt (filter by my record IDs) was blocked by the permission classifier, so that is the user's call. `docker images` shows no click/swerebench/`hb__` image and no container remains.

Estimate for the 16-candidate list (D from one sample, ratio on-disk 3.27 GB / compressed 1.24 GB = 2.6x, G for other repos): compressed total 24.5 GB (`data/calibration-16.json`), so **about 65 GB retained** if nothing is pruned between tasks; largest image 2.15 GB compressed, about 5.7 GB peak. Either keep >= 70 GB free (the 70.1 GB reclaimable local volumes would do) or accept a prune of BuildKit cache between passes (`docker builder prune` is the user's decision; it also drops other users' cache). `gpu_swe_fresh.sh` refuses to start a stage below `MIN_FREE_GB=30` free on `/mnt/scratch` (gate `disk-low-<n>GB`, exit 5). Per-task Harbor trial directories are small (about 350 KB with verifier output).

## Selection (CPU, deterministic, already run)

```bash
cd ~/repos/swe-fresh
curl -sSL -o data/rebench-2026_03.parquet \
  https://huggingface.co/api/datasets/nebius/SWE-rebench-leaderboard/parquet/default/2026_03/0.parquet   # sha256 18e198ac... recorded
python3 select_candidates.py --parquet data/rebench-2026_03.parquet --sizes data/sizes-2026_03.json \
  --seed swe-fresh-calibration-20261004 --count 16 --out data/calibration-16.json   # drop --sizes to refetch Docker Hub sizes
uvx harbor==0.23.0 download swe-rebench/swe-rebench-leaderboard -o data/dl                  # 860 tasks, 32 s, snapshot used by -p
```

Pool 110 -> 76 eligible (patch <= 150 lines and <= 3 files, statement <= 6,000 chars, image <= 2.5 GB compressed; `harbor_memory` is null for all 110) -> 60 repos -> 16 by seeded hash. The list (`data/calibration-16.json`, test `test_recorded_split_reproduces_the_frozen_calibration_list`): pycqa__isort-2491, delgan__loguru-1451, stanfordnlp__dspy-9478_interface, getmoto__moto-9841, beever-ai__beever-atlas-102, cs-si__eodag-2176, meltano__meltano-9950, pandas-dev__pandas-64796, geopython__pygeoapi-2338, agronholm__anyio-1134, macbre__sql-metadata-625_interface, graphistry__pygraphistry-1277, reframe-hpc__reframe-3660_interface, psd-tools__psd-tools-593, conan-io__conan-19750, huggingface__huggingface_hub-4056. Three are `_interface` variants (the task adds an interface description). Not filtered: tests that need network or a GPU at test time (G: calibration solved-2/2 removes them). Canary: `pallets__click-3239` (smallest image, not in the list).

## Pins and draft manifests (generated; parent reviews, submits)

```bash
export CUDA_VISIBLE_DEVICES=""
python3 make_manifest.py pin --engine hq --tasks-file data/calibration-16.json \
  --dataset-path data/dl/swe-rebench-leaderboard --canary-task pallets__click-3239 --pins pins/hq-cal-v1.json    # exclusive-create; same for llama
python3 make_manifest.py verify --pins pins/hq-cal-v1.json
python3 make_manifest.py manifest --pins pins/hq-cal-v1.json --stage canary  --out results/cal-hq-v1 \
  --xid-since "2026-10-03 18:57:00" --canary-task pallets__click-3239 --id swe-fresh-cal-hq-canary  --runs 2 > <SP>/overnight/items/swe-fresh-cal-hq-canary.toml
python3 make_manifest.py manifest --pins pins/hq-cal-v1.json --stage measure --out results/cal-hq-v1 \
  --xid-since "2026-10-03 18:57:00" --canary-task pallets__click-3239 --id swe-fresh-cal-hq-measure --runs 2 > <SP>/overnight/items/swe-fresh-cal-hq-measure.toml
```

Delivered: `<SP>/overnight/items/swe-fresh-cal-{llama,hq}-{canary,measure}.toml` (SP = the scratchpad of CONTEXT.md). Same OUT for canary and measure, as in `two-agent/codex/gpu_ab.sh` (measure requires the passed canary and equal pins hash). `XID_SINCE` is the value used by the other items; the parent must confirm it. Pins cover code, test, stub, template, engine binaries/model (llama: stack2 build + IQ4_XS gguf + draft vocab; HQ: overlay, vllm sources, codexh checkpoint) and the dataset-snapshot files of the 17 tasks; editing any of them needs new pins. Engine choice is the parent's: HQ codexh k5 (`MAX_SEQS 1`, `MAX_LEN 102400`, `GPU_UTIL 0.86`) or llama stack2 IQ4_XS S5 (`-np 1 -c 102400`); both are copies of the `gpu_ab.sh` launch lines with the slot count parameterised.

Canary PASS: server healthy, 1 task through Harbor + relay, status not `invalid` (unsolved/timeout are acceptable), >= 2 completed model calls, >= 1 tool call, exact engine counters present (llama `timings` or HQ `vllm:` deltas). FAIL: `no_model_calls` (provider route broken), call failures, Harbor exception, counters unavailable, budget. The canary does not require a solve (a 27B at 300 s may fail click-3239).

## GPU time (D from M overheads, G for the agent phase)

Per task-run, the GPU idles during image + container + Codex install: about 104-127 s install (M, 3 runs) + 3 s env when the image is cached, +46 s on first pull (M, 1.24 GB), + 2-4 s verifier on click (G: minutes for large suites such as pandas). The agent phase is capped at 1200 s; REPORT's guess is 8-15 min average.

| Plan | Runs | Estimate | Ceiling |
|---|---|---|---|
| Canary (one boot) | 1 | 13 min llama / 25 min HQ (boot 3 / 15 min) | 33 / 45 min budget |
| Pass of 16 (one run each) | 16 | about 3.8 h at 10 min average agent (est in manifest with `--runs 1`) | 6.6 h if every run hits the cap |
| Report's design, 16 x 2 in one item (`--runs 2`) | 32 | about 7.5 h (manifest est 7.5 h llama, 7.7 h HQ) | 13 h budget |

This is above the REPORT's 6-7 h because that figure omits the about 2.5 min per run of Codex install and container setup (about 80 min over 32 runs, GPU idle). Recommended: pass A with `--runs 1` for all 16 (about 4 h), then a second item on only the tasks solved in A with wall 5-15 min (a second tasks file, fresh pins and OUT, canary again about 15 min), which applies the REPORT's "solved 2/2" rule without paying for second runs of failures. Round-major order inside an item keeps drift from hitting one task. Not implemented: prebaking Codex into an image or overlapping the next task's setup with the current agent phase (would remove the idle share).

Memory: each container is limited to 8 GB by the task (850 of 860 tasks; 10 have 16 GB); with HQ resident the 31 GB host can run one at a time, so the manifests use one agent. Pull bandwidth: about 24.5 GB compressed total (Docker Hub, free).

## Selection after calibration

REPORT 2.2 step 3 unchanged: keep tasks solved in all runs with wall 5-15 min and context high water >= 50k (`probe.peak_context_tokens`, relay input+output max) on at least one run; pick 6 over >= 4 repos; freeze `rb6.json` with ids, image digests (`data/calibration-16.json` has them), the dataset ref (`probe.dataset_ref`) and calibration numbers. Per-run facts: `SUMMARY.json.per_task[]` (status, scored wall, peak context, calls, tokens, decode tok/s, acceptance, compactions), job files for per-call stats (`call_stats.by_depth`, TTFT, cache hit, 500s, runaways), raw `relay/model-calls.jsonl` and the Codex rollout (`agent/sessions`) under `attempts/`.

## Judge fixes (2026-10-04, after the Opus review; pins and items regenerated as `*-cal-v2`)

- gpuq items carry `PATH=~/.local/bin:/usr/local/bin:/usr/bin:/bin` (uvx is not on gpuq's default PATH) and `exit_codes` 3 hazard, 4 transient, 5 deterministic, 6 integrity, 7 timeout (gpuq's classes; unmapped, rc 5 and 7 would read as env_ram and hazard). The identity failure text is now `pinned inputs changed`, which also matches gpuq's integrity regex; the exit code carries the class either way. Every `uvx` call is `harbor==0.23.0`.
- Disk floor: canary 5 GB, measure 30 GB at stage start, plus a per-task check before each task starts (`--min-free-gb`, free space of `/mnt/scratch`). A stop is the gate `disk_low: <free> GiB < <min> GiB before <task>`, runner exit 5, status `<stage>-disk-low` (exit 5, deterministic).
- A timed-out task whose only failed relay calls were in flight at the cap (started within 30 s of `agent_execution.started_at + cap`, or ended in a client disconnect that finished after that point) scores `timeout`, whatever the verifier says (the first real canary had reward 1 after the kill and was wrongly `invalid`; fixed 2026-10-04, see below).
- `response.incomplete` with `incomplete_details.reason` in {max_output_tokens, max_tokens, max_new_tokens, length}, or without a reason and >= 32768 output tokens, is no longer a failed call: it counts as a completed call, sets `probe.runaway` and `call_stats.truncated_calls`; the task keeps its score by the usual rules. The reason strings are inferred (no recorded fixture of a real engine's incomplete event exists).
- Abnormal stop: the runner handles SIGINT/SIGTERM, signals the Harbor process group it recorded (`harbor.process.json`), then `docker compose --project-name <trial>__env down --volumes --remove-orphans` for the project names recorded from the job's trial directories (`compose-projects.json`). `run_swe_fresh.py cleanup --campaign DIR` is the backstop after a SIGKILL of the runner (kills a recorded group only if pid and /proc start tick match) and `gpu_swe_fresh.sh` calls it. Tested with a fake Harbor child; the compose call is mocked (never run in tests). A `docker compose down` of a nonexistent project works without compose files (probed once: warning, rc 0).
- `egress_scan.py` scans saved rollouts' call arguments (function_call arguments, custom tool input, shell argv) for pip/uv/git network commands, curl, wget, github.com, githubusercontent, pypi.org, `git log --all`. It runs per task (`probe.egress_scan`), `summarize --rescan-egress` rewrites it from disk, and SUMMARY lists `egress_flagged_tasks`. A hit is a flag to review (a legitimate `pip install -e .` also matches), not proof of leakage; web search itself is disabled in the Codex config.

## Real canary findings (2026-10-04 02:55, llama, 300 s cap; `fixtures/cal-llama-canary`)

- Root cause of `invalid: call_failed [53]`: the in-flight exemption applied only `if timed_out and not solved`; Harbor runs the verifier after the kill (reward 1 here), so `solved` was true and the cut call counted as a failure. The exemption now applies to every timed-out run.
- Harbor's cap kills only the host `docker compose exec` client; Codex kept calling the model for about 18 s (calls 50-53 ended at +305 s, +311 s, +316 s, +318 s; the verifier started at +305 s; call 53 was cut by the teardown). The relay now takes `--cap-s` and answers 403 (`post_cap`, never forwarded, no GPU time) to a `/responses` call starting after the first call plus the cap (the agent start is not known live; the first call follows it by about 1 s). Scoring uses Harbor's exact `agent_execution.started_at + cap`: calls starting later are outside the scored statistics (`call_stats.post_cap_calls`, `post_cap_served_calls`), and a timed-out run whose verifier reward could include post-cap work (a served post-cap model call, or a rollout tool call stamped after the cap) is `timeout` with `probe.post_cap.reward_not_credited`; SUMMARY lists `post_cap_activity_tasks`. A timed-out run with no such activity stays `passed`, its scored wall clamped to the cap and `censored`. Engine counters (`metrics_delta`) still include any post-cap generation.
- gpuq's fresh retry renames OUT to `OUT.void-a<N>` before the stage starts, which took the canary proof with it. The measure stage now runs `run_swe_fresh.py canary-proof` (newest valid, pin-matching canary among OUT and `OUT.void-a*`) and copies it to `OUT/canary-proof/`; no manifest change.
- Egress verdict for that canary: all 11 hits were `github.com` text inside Codex's own Python edit scripts (issue links already in click's `core.py` comments, and one docstring), no pip/curl/wget/git-network command. The scan missed `git log --oneline --all` (pattern was `git log --all`); fixed. The agent did read local history (`git log --all`, `git show 6c4a77b`); HEAD was the newest commit, so no later fix was visible.
