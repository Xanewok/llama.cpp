"""Regression tests for the judged defects D1-D9 and the surviving mutants. Each test names what it pins."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from scripts.opt import cli, configs, fit, goal as goalmod, report, workload as W
from scripts.opt.ingest import campaign, classify_dir, kld, queue_np, run, two_agent
from scripts.opt.ingest.base import measurement
from scripts.opt.store import Reject, Store

from .synth import PROV, add_cfg, add_m, llama_cmd
from .test_report import GOAL, kld_row, llama_candidate, render, world
from .test_workload import PROFILE, synthetic_campaigns

REPO = Path(__file__).resolve().parents[2]
SWIFT = "Swift-1.5-Qwen3.8-27B-"


def swift_llama(store, recipe, decode, env=None, kv="q8_0", prefill=800.0, free=2000.0, np_=1, ctx=102400, stack="s", ver="1.5"):
    stem = SWIFT.replace("1.5", ver) if ver != "1.5" else SWIFT
    cfg = add_cfg(store, llama_cmd(stem + recipe, np_, ctx, kv), stack=stack, env=env)
    for metric, v in (("decode_tok_s", decode), ("prefill_tok_s", prefill), ("output_tokens", 100000.0),
                      ("prefill_new_tokens", 100000.0)):
        add_m(store, cfg, metric, v, item=recipe + str(env) + ver)
    add_m(store, cfg, "vram_min_free_mib", free, gpu=0, item=recipe + str(env) + ver)
    return cfg


def base_world(tmp_path):
    s = Store(tmp_path)
    synthetic_campaigns(s, n=3)
    s.upsert("workloads", {**PROFILE, "id": "profile:codex-8task-1agent", "kind": "profile", "n_campaigns": 1,
                           "sources": ["x"], "workload_id": "codex-8task-1agent"})
    return s


# ---- D1: candidate identity -----------------------------------------------------------------

def test_d1_model_recipes_with_one_quant_label_are_separate_candidates(tmp_path):
    s = base_world(tmp_path)
    swift_llama(s, "IQ4_XS", 100.0)
    swift_llama(s, "IQ4_XS-ssmout6", 90.0)
    swift_llama(s, "J4-bf16", 80.0)
    swift_llama(s, "J4-fromQ8", 85.0)
    cands = [c for c in report.candidates(s, 1) if c["engine"] == "llama"]
    assert len(cands) == 4


def test_d1_ab_env_arms_are_never_pooled(tmp_path):
    s = base_world(tmp_path)
    swift_llama(s, "IQ4_XS", 100.0, env={"GGML_CUDA_FA_MMA_Q8_0": 1})
    swift_llama(s, "IQ4_XS", 70.0, env={"GGML_CUDA_FA_MMA_Q8_0": 0})
    swift_llama(s, "IQ4_XS", 100.0, env={"GGML_CUDA_FA_VEC_QUANT_MAX_BATCH": 2})
    assert len([c for c in report.candidates(s, 1) if c["engine"] == "llama"]) == 3


def test_d1_flag_variants_are_separate_candidates(tmp_path):
    s = base_world(tmp_path)
    a = llama_cmd(SWIFT + "IQ4_XS") + ["--spec-draft-vocab", "swift_draft_vocab_ids.json"]
    b = llama_cmd(SWIFT + "IQ4_XS") + ["--tensor-split", "69,31"]
    for cmd, i in ((llama_cmd(SWIFT + "IQ4_XS"), "plain"), (a, "vocab"), (b, "split")):
        cfg = add_cfg(s, cmd)
        add_m(s, cfg, "decode_tok_s", 100.0, item=i)
        add_m(s, cfg, "prefill_tok_s", 800.0, item=i)
    assert len([c for c in report.candidates(s, 1) if c["engine"] == "llama"]) == 3


def test_d1_hq_versions_with_one_checkpoint_name_are_separate(tmp_path):
    s = base_world(tmp_path)
    for ver in ("Swift-1.5-Qwen3.8-27B", "Swift-Qwen3.8-27B"):
        cfg = add_cfg(s, ["python", "vllm", "serve", f"/m/{ver}-W4A16-syv-fast", "--max-num-seqs", "1", "--max-model-len", "102400",
                          "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.86",
                          "--speculative-config", '{"num_speculative_tokens": 5}'])
        add_m(s, cfg, "decode_tok_s", 100.0, item=ver)
        add_m(s, cfg, "prefill_tok_s", 800.0, item=ver)
    hq = [c for c in report.candidates(s, 1) if c["engine"] == "hq" and "syv-fast" in c["model_id"]]
    assert sorted(c["model_id"] for c in hq) == ["swift-1.0:W4A16-syv-fast", "swift-1.5:W4A16-syv-fast"]


def test_d1_model_identity_parses_version_and_recipe():
    mi = configs.model_identity
    assert mi("/x/Swift-1.5-Qwen3.8-27B-IQ4_XS-ssmout6.gguf")[0] == "swift-1.5:IQ4_XS-ssmout6"
    assert mi("/x/Swift-Qwen3.8-27B-IQ4_XS.gguf")[0] == "swift-1.0:IQ4_XS"
    assert mi("/x/Qwen3.5-4B-UD-Q4_K_XL.gguf")[0].startswith("other:")


def test_d1_goal_model_filter_excludes_other_models(tmp_path):
    s = base_world(tmp_path)
    swift_llama(s, "IQ4_XS", 100.0)
    swift_llama(s, "IQ4_XS", 150.0, ver="1.0")
    text = GOAL.replace("[constraints]", '[constraints]\nmodel = ["swift-1.5:*"]')
    _, ev = report.render(s, goalmod.parse(text), show_fail=True)
    assert {r["cand"]["model_id"] for r in ev["rows"] if r["cand"]["engine"] == "llama"} == {"swift-1.5:IQ4_XS"}
    assert ev["excluded"]
    with pytest.raises(goalmod.GoalError):
        goalmod.parse(GOAL.replace("[constraints]", "[constraints]\nmodel = []"))


def test_d1_shipped_goals_are_swift_1_5_only():
    for f in sorted((REPO / "results/db/goals").glob("*.toml")):
        assert goalmod.load(f)["constraints"]["model"] == ["swift-1.5:*"], f.name


# ---- D2: KLD join ---------------------------------------------------------------------------

def refbf_json(path, arms):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"refs": {}, "kld": {a: {"REFBF": {"all_prefill": [v, 0.0003, [1, 2, 3, 4, 5, 6]]}}
                                                    for a, v in arms.items()}}))
    return path


def ingest_kld(tmp_path, arms):
    p = refbf_json(tmp_path / "kld" / "results.json", arms)
    s = Store(tmp_path / "db")
    b = kld.parse(p)
    for c in b.configs.values():
        s.upsert("configs", c)
    for m in b.measurements:
        s.upsert("measurements", m)
    return s


def test_d2_j4_fromq8_does_not_receive_j4_bf16_kld(tmp_path):
    s = ingest_kld(tmp_path, {"loc_j4bf16": 0.00307, "loc_j4q8": 0.00319})
    base = base_world(tmp_path / "w")
    for c in s.tables["configs"].values():
        base.upsert("configs", c)
    for m in s.rows("measurements"):
        base.upsert("measurements", m)
    swift_llama(base, "J4-fromQ8", 90.0, kv="f16")
    swift_llama(base, "J4-bf16", 90.0, kv="f16")
    _, ev = render(base)
    got = {r["cand"]["model_id"]: r["kld"]["kld@REFBF"] for r in ev["rows"] if r["cand"]["engine"] == "llama"}
    assert got["swift-1.5:J4-fromQ8"][0] == pytest.approx(0.00319)
    assert got["swift-1.5:J4-bf16"][0] == pytest.approx(0.00307)


def test_d2_no_fallback_to_another_kv(tmp_path):
    s = ingest_kld(tmp_path, {"loc_j4bf16": 0.00307})  # dumped at f16 KV
    base = base_world(tmp_path / "w")
    for c in s.tables["configs"].values():
        base.upsert("configs", c)
    for m in s.rows("measurements"):
        base.upsert("measurements", m)
    swift_llama(base, "J4-bf16", 90.0, kv="q8_0")
    _, ev = render(base)
    r = next(r for r in ev["rows"] if r["cand"]["engine"] == "llama")
    assert r["kld"]["kld@REFBF"] is None and r["verdict"] == "UNKNOWN"
    assert r["kld_other"]["kld@REFBF"][0][0] == "f16/f16"  # shown as unverified context only


def test_d2_arm_table_is_explicit_not_fuzzy():
    assert kld.arm_identity("fdown6mid")[0] == "swift-1.5:fdown6mid-bf16"
    assert kld.arm_identity("loc_j4q8")[:2] == ("swift-1.5:J4-fromQ8", "f16/f16")  # "q8" is the recipe, not the KV
    assert kld.arm_identity("J5q_q8")[0] is None  # not in the table: never guessed
    assert kld.arm_identity("something_new")[0] is None


# ---- D3: missing KLD never outranks verified; ledger-prior is context ------------------------

def test_d3_unverified_kld_ranks_below_every_verified_candidate(tmp_path):
    s = base_world(tmp_path)
    fast = swift_llama(s, "J5q-bf16", 150.0)   # fast, no KLD at all
    slow = swift_llama(s, "J4-bf16", 60.0)     # slow, KLD verified but VRAM unmeasured below
    from scripts.opt.ingest.kld import weights_config
    c = weights_config("swift-1.5:J4-bf16", "q8_0/q8_0")
    s.upsert("configs", c)
    s.upsert("measurements", measurement(source="kld", metric="kld@REFBF", value=0.002, unit="nats", config_id=c["config_id"],
                                         workload_id="kld:x", prov=PROV, dims={"bucket": "all_prefill", "arm": "x"}, item_id="k", ident="k"))
    for mid in list(s.tables["measurements"]):
        m = s.tables["measurements"][mid]
        if m["config_id"] == slow["config_id"] and m["metric"] == "vram_min_free_mib":
            del s.tables["measurements"][mid]
    _, ev = render(s)
    rows = [r for r in ev["rows"] if r["cand"]["engine"] == "llama"]
    assert [r["cand"]["model_id"] for r in rows] == ["swift-1.5:J4-bf16", "swift-1.5:J5q-bf16"]
    assert rows[0]["wall_est"] > rows[1]["wall_est"] and rows[0]["verdict"] == rows[1]["verdict"] == "UNKNOWN"


PLAN_LINES = """\
1. **Fidelity** (KLD vs Swift 1.5 Q6_K, 30k assistant/doc rows, contexts <= 8k): Q5_K_M 0.0040, IQ4_XS 0.0109, **HyperQwen Swift 1.5
   W4A16 0.0254** (2.3x IQ4_XS; includes vLLM numerics), HQ 1.0 0.0353. KV cache format irrelevant.
- **10:45 goal matrix**: KLD W4A16 g128 0.0254, g32 + int8 attn 0.0159, mixed 0.0133 (does not fit 2 x 100k) vs IQ4_XS 0.0109.
- **13:35 V5 done**: best `qkvo_codexh` = mixed32_qkvo scheme -> KLD 0.0134 (M), paired
"""


def test_d3_ledger_prior_rows_have_their_own_reference_and_never_pass(tmp_path):
    from scripts.opt.ingest import ledger
    plan = tmp_path / "PLAN.md"
    plan.write_text(PLAN_LINES)
    b = ledger.parse(plan)
    assert {m["metric"] for m in b.measurements} == {"kld@ledger-q6k-8k"}
    assert {m["status"] for m in b.measurements} == {"ledger-prior"}
    vals = {b.configs[m["config_id"]]["model_id"]: m["value"] for m in b.measurements}
    assert vals["swift-1.5:IQ4_XS"] == 0.0109 and vals["swift-1.5:W4A16-syv-fast"] == 0.0254
    assert vals["swift-1.5:W4A16-g32-attn-int8-codexh"] == 0.0134 and vals["swift-1.0:W4A16-syv-fast"] == 0.0353
    # as report context: shown, never a verdict source
    s = base_world(tmp_path / "w")
    for c in b.configs.values():
        s.upsert("configs", c)
    for m in b.measurements:
        s.upsert("measurements", m)
    swift_llama(s, "IQ4_XS", 100.0)
    text, ev = render(s)
    r = next(r for r in ev["rows"] if r["cand"]["engine"] == "llama")
    assert r["verdict"] == "UNKNOWN" and r["kld"]["kld@REFBF"] is None
    assert r["ledger"] and r["ledger"][0][0] == 0.0109 and "ledger KLD" in text and "not REFBF" in text


def test_d3_ledger_entry_that_left_the_plan_is_dropped_not_invented(tmp_path):
    from scripts.opt.ingest import ledger
    plan = tmp_path / "PLAN.md"
    plan.write_text("Q5_K_M 0.0040, IQ4_XS 0.0109\n")
    b = ledger.parse(plan)
    assert {b.configs[m["config_id"]]["model_id"] for m in b.measurements} == {"swift-1.5:IQ4_XS", "swift-1.5:Q5_K_M"}
    assert len(b.flagged) == 5


# ---- D4: calibration honesty ----------------------------------------------------------------

def test_d4_spearman_uses_average_ranks_for_ties():
    assert W._rank_corr([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660254, abs=1e-6)


def two_config_campaigns(store, speeds):
    """cfg per k; runs[i] = (k, prefill, decode). Call durations obey the cost model with each run's own speeds."""
    tasks = ["rust/doubly-linked-list", "rust/book-store", "rust/react", "rust/alphametics", "rust/forth", "rust/xorcism",
             "rust/nucleotide-codons", "javascript/complex-numbers"]
    cfgs = {}
    for i, (k, pf, dc) in enumerate(speeds):
        if k not in cfgs:
            cfgs[k] = add_cfg(store, ["python", "vllm", "serve", f"/m/{SWIFT}W4A16-codexh", "--max-num-seqs", "1",
                                     "--max-model-len", "102400", "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.86",
                                     "--speculative-config", '{"num_speculative_tokens": %d}' % k])
        cfg = cfgs[k]
        tdict, wall = {}, 0.0
        for t in tasks:
            calls = []
            for j in range(3 + i % 2):
                new, out = 2000 + 100 * j, 300 + 40 * j
                calls.append([10000 + 3000 * j, new, out, 1 if j else 0, 1.5 if j else 0.0, 0.1 + new / pf + out / dc])
            tdict[t] = calls
            tw = sum(c[5] + c[4] for c in calls) + 5.0
            wall += tw
            add_m(store, cfg, "task_wall_s", tw, item=f"r{i}", sha=f"sha{i}", task=t)
        for metric, v in (("prefill_tok_s", pf), ("decode_tok_s", dc), ("wall_s", wall)):
            add_m(store, cfg, metric, v, item=f"r{i}", sha=f"sha{i}")
        store.upsert("workloads", {"id": f"calls:r{i}", "kind": "campaign-calls", "campaign": f"r{i}", "config_id": cfg["config_id"],
                                   "summary_sha256": f"sha{i}", "workload_id": "codex-8task-1agent", "valid": True,
                                   "tasks": tdict, "wall_s": wall})


def tier(res, prefix):
    return next((v for k, v in res["tiers"].items() if k.startswith(prefix)), "missing")


def test_d4_tier_b_is_held_out_needs_a_replicate_and_never_uses_the_runs_own_speeds(tmp_path):
    s = Store(tmp_path)
    # four configs, one run each: no replicate exists, so no held-out estimate exists
    two_config_campaigns(s, [(3, 1000.0, 100.0), (4, 1000.0, 100.0), (5, 1000.0, 100.0), (6, 1000.0, 100.0)])
    assert tier(W.calibrate(s), "B (held out)") is None
    # replicate of k=3 with a different decode speed: the held-out estimate takes the replicate's speed
    s2 = Store(tmp_path / "2")
    two_config_campaigns(s2, [(3, 1000.0, 100.0), (3, 1000.0, 200.0), (4, 1000.0, 100.0), (5, 1000.0, 100.0), (6, 1000.0, 100.0)])
    res = W.calibrate(s2)
    held = {r[0]: r for r in res["rows"]["B"]}
    in_sample = {r[0]: r for r in res["rows"]["B0"]}
    assert held["r0"][3] < in_sample["r0"][3] - 10.0   # the replicate decodes 2x faster: the held-out estimate is far lower
    assert "A (in-sample)" in "".join(res["tiers"]) and "B0 (in-sample)" in "".join(res["tiers"])


def test_d4_rank_correlation_is_reported_per_tier(tmp_path):
    s = Store(tmp_path)
    two_config_campaigns(s, [(3, 1000.0, 100.0), (3, 1000.0, 200.0), (4, 1000.0, 100.0), (5, 1000.0, 120.0), (6, 1000.0, 90.0)])
    res = W.calibrate(s)
    assert {"A", "B0", "B"} <= set(res["rank_corr"])
    assert "Spearman" in W.report_text(res)


def test_d4_rho_is_labelled_as_fitted_on_two_runs():
    assert "n=2" in W.RHO_SOURCE
    assert W.RHO == 0.0


def with_kld(s, recipes, kv="q8_0/q8_0"):
    from scripts.opt.ingest.kld import weights_config
    for rcp in recipes:
        c = weights_config(f"swift-1.5:{rcp}", kv)
        s.upsert("configs", c)
        s.upsert("measurements", measurement(source="kld", metric="kld@REFBF", value=0.002, unit="nats", config_id=c["config_id"],
                                             workload_id="kld:x", prov=PROV, dims={"bucket": "all_prefill", "arm": rcp},
                                             item_id=rcp, ident=rcp))


def test_d4_candidates_inside_the_held_out_error_are_one_tie_band(tmp_path):
    s = base_world(tmp_path)
    for rcp, d in (("A-bf16", 100.0), ("B-bf16", 101.0), ("C-bf16", 20.0)):
        swift_llama(s, rcp, d)
    with_kld(s, ["A-bf16", "B-bf16", "C-bf16"])
    text, ev = render(s)
    rows = {r["cand"]["model_id"].split(":")[1]: r for r in ev["rows"] if r["cand"]["engine"] == "llama"}
    assert rows["A-bf16"]["band"] == rows["B-bf16"]["band"] != rows["C-bf16"]["band"]
    assert "tie band" in text and "measured wall" in text


def test_d4_measured_wall_is_shown_next_to_the_estimate(tmp_path):
    s = base_world(tmp_path)
    cfg = swift_llama(s, "A-bf16", 100.0)
    add_m(s, cfg, "wall_s", 1234.0, item="w1", sha="w1")
    with_kld(s, ["A-bf16"])
    text, ev = render(s)
    r = next(r for r in ev["rows"] if r["cand"]["engine"] == "llama")
    assert r["cand"]["measured_wall"] == (1234.0, 1) and "1234 m (n=1)" in text


def test_d4_f1_basis_and_measured_basis_are_never_ranked_in_one_list(tmp_path):
    s = base_world(tmp_path)
    slow = swift_llama(s, "SLOWMEAS-bf16", 40.0)                  # campaign-agg speeds: measured basis, slow
    fast = add_cfg(s, llama_cmd(SWIFT + "FASTF1-bf16", 1, 102400, "q8_0"), stack="s")
    for depth in (8192, 32768, 98304):                           # F1 basis only (queue-np): fast
        add_m(s, fast, "decode_tok_s", 300.0, source="queue-np", workload="synthetic:steady-decode", item="q", np=1, depth=depth)
        add_m(s, fast, "prefill_tok_s", 3000.0, source="queue-np", workload="synthetic:steady-decode", item="q", np=1, depth=depth)
    add_m(s, fast, "vram_min_free_mib", 2000.0, gpu=0, item="q")   # same verdict as the slow row, so only the class can separate them
    with_kld(s, ["SLOWMEAS-bf16", "FASTF1-bf16"])
    text, ev = render(s)
    rows = [r for r in ev["rows"] if r["cand"]["engine"] == "llama"]
    assert [r["verdict"] for r in rows] == ["PASS", "PASS"]
    assert [r["cand"]["speeds"]["basis"] for r in rows] == ["campaign-agg", "F1"]
    assert rows[1]["wall_est"] < rows[0]["wall_est"]            # the F1 row is faster yet listed after, in its own list
    assert text.count("## Ranked") == 2 and "never comparable" in text


# ---- D5: solve_rate -------------------------------------------------------------------------

def test_d5_solve_rate_bound_is_evaluated_from_campaign_outcomes(tmp_path):
    s = base_world(tmp_path)
    good, bad = swift_llama(s, "GOOD-bf16", 100.0), swift_llama(s, "BAD-bf16", 100.0)
    add_m(s, good, "solve_rate", 1.0, item="g", sha="g")
    add_m(s, bad, "solve_rate", 0.5, item="b", sha="b")
    with_kld(s, ["GOOD-bf16", "BAD-bf16"])
    _, ev = render(s, GOAL + "solve_rate = { min = 0.9 }\n")
    v = {r["cand"]["model_id"].split(":")[1]: r for r in ev["rows"] if r["cand"]["engine"] == "llama"}
    assert v["BAD-bf16"]["checks"]["solve_rate"] is False and v["BAD-bf16"]["verdict"] == "FAIL"
    assert v["GOOD-bf16"]["checks"]["solve_rate"] is True and v["GOOD-bf16"]["verdict"] == "PASS"


# ---- D6: campaigns with tasks that did not run -----------------------------------------------

def test_d6_campaign_with_a_not_run_task_is_ingested_with_the_failure_reason(fx, db):
    d = fx / "campaign-llama-lost4"
    job = sorted(d.glob("job-*.json"))[0]
    task = json.loads(job.read_text())["task"]
    job.write_text(json.dumps({"status": "failed", "model": "m", "task": task, "attempt": "attempts/x",
                               "reason": "vram_reserve: GPU-0", "finished_at": 1.0}))
    Path(str(job) + ".sha256").unlink()
    rep = run([d], db=db)
    assert not rep["rejected"]
    st = Store(db)
    solved = [m for m in st.rows("measurements") if m["metric"] == "task_solved" and m["dims"]["task"] == task]
    assert len(solved) == 1 and solved[0]["value"] == 0 and "vram_reserve" in solved[0]["reason"]
    assert solved[0]["dims"]["ran"] is False and solved[0]["status"] == "invalid"
    assert {m["status"] for m in st.rows("measurements") if m["metric"] == "wall_s"} == {"invalid"}


# ---- D7: ids from content --------------------------------------------------------------------

KLD_TXT = """== KL(REF6 || arm), REF_q6k_f16.kldo; top-64; assistant rows; n_seq = 6; SE = sequence-level ==

arm            +GB MiB/GPU      KLD    seqSE |   d vs J4      ±se  neg      d/GB
J4          +0.000      +0  0.00337  0.00031 | control
J5q         +0.895    +427  {kld}  0.00021 |  -0.00105  0.00010 6/6  -0.00117
"""


def test_d7_two_kld_files_in_same_named_directories_do_not_collide(tmp_path):
    for sub, v in (("a", "0.00232"), ("b", "0.00251")):
        p = tmp_path / sub / "out-b1" / "results.txt"
        p.parent.mkdir(parents=True)
        p.write_text(KLD_TXT.format(kld=v))
    run([tmp_path / "a", tmp_path / "b"], db=tmp_path / "db")
    vals = sorted(m["value"] for m in Store(tmp_path / "db").rows("measurements") if m["metric"] == "kld@REF6" and m["dims"]["arm"] == "J5q")
    assert vals == [0.00232, 0.00251]


def test_d7_identical_content_at_two_paths_is_one_row_with_both_paths(tmp_path):
    for sub in ("a", "b"):
        p = tmp_path / sub / "out-b1" / "results.txt"
        p.parent.mkdir(parents=True)
        p.write_text(KLD_TXT.format(kld="0.00232"))
    run([tmp_path / "a", tmp_path / "b"], db=tmp_path / "db")
    rows = [m for m in Store(tmp_path / "db").rows("measurements") if m["metric"] == "kld@REF6" and m["dims"]["arm"] == "J5q"]
    assert len(rows) == 1 and rows[0]["provenance"].get("also_at")


def test_d7_same_named_campaign_dirs_with_different_content_keep_separate_call_tables(fx, tmp_path):
    for sub, label in (("a", "x"), ("b", "y")):
        dst = tmp_path / sub / "campaign-x"
        shutil.copytree(fx / "campaign-hq-lost4", dst)
        sm = json.loads((dst / "SUMMARY.json").read_text())
        sm["note"] = label
        (dst / "SUMMARY.json").write_text(json.dumps(sm))
    run([tmp_path / "a", tmp_path / "b"], db=tmp_path / "db")
    st = Store(tmp_path / "db")
    calls = [w for w in st.rows("workloads") if w.get("kind") == "campaign-calls"]
    assert len(calls) == 2 and len({m["item_id"] for m in st.rows("measurements") if m["metric"] == "wall_s"}) == 2


# ---- D8: VRAM bands, HQ pool per slot, prefix rule --------------------------------------------

def test_d8_derived_vram_inside_the_loo_error_band_needs_a_probe():
    assert fit.classify(600) == "fits"                 # a measured value classifies as is
    assert fit.classify(600, band=528) == "probe-needed"
    assert fit.classify(1100, band=528) == "fits"
    assert fit.classify(-400, band=100) == "reject"


def test_d8_loo_band_is_the_p90_residual():
    model = {"loo": [{"resid": r} for r in (100, -200, 300, -400, 500, 600, -700, 800, -900, 6000)]}
    assert fit.loo_band(model) == 900
    assert fit.loo_band({"loo": []}) == fit.PROBE_MIB


def hq_candidate_world(tmp_path, pool, np_=1):
    s = base_world(tmp_path)
    cfg = add_cfg(s, ["python", "vllm", "serve", f"/m/{SWIFT}W4A16-codexh", "--max-num-seqs", str(np_), "--max-model-len", "102400",
                      "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.86",
                      "--speculative-config", '{"num_speculative_tokens": 5}'])
    for metric, v in (("decode_tok_s", 100.0), ("prefill_tok_s", 800.0), ("output_tokens", 100000.0), ("prefill_new_tokens", 100000.0)):
        add_m(s, cfg, metric, v, item="hq")
    add_m(s, cfg, "kv_pool_tokens", pool, source="hq-conc", item="hqboot", sha="hqboot")
    add_m(s, cfg, "vram_min_free_mib", 900.0, gpu=0, item="hq")
    return s


def test_d8_hq_fits_checks_pool_per_slot_not_only_max_model_len(tmp_path):
    _, ev = render(hq_candidate_world(tmp_path, pool=90000))            # 90k tokens for np=1 < 98304
    r = next(r for r in ev["rows"] if r["cand"]["engine"] == "hq")
    assert r["checks"]["fits"] is False and any("KV pool per slot 90k (measured)" in n for n in r["notes"])
    _, ev = render(hq_candidate_world(tmp_path / "ok", pool=199752))
    assert next(r for r in ev["rows"] if r["cand"]["engine"] == "hq")["checks"]["fits"] is True


def test_d8_prefix_rule_is_labelled_in_sample(fx, db):
    run([fx / "hq-conc"], db=db)
    text, _ = fit.report(Store(db))
    assert "IN-SAMPLE" in text


def test_d8_prefix_depth_is_rounded_up_to_whole_blocks():
    assert fit.hq_prefix_state(1000, 300) == "kept"
    assert fit.hq_prefix_state(1000, 300, unit=64) == "lost"     # 300 -> 320 tokens of pool = 0.32


def test_d8_kv_pool_of_one_boot_is_one_observation(fx, db):
    run([fx / "hq-conc"], db=db)
    st = Store(db)
    pools = fit.fit_hq(st)["pools"]
    assert len({(p["item"], p["config_id"], p["pool"]) for p in pools}) == len(pools)
    raw = [m for m in st.rows("measurements") if m["metric"] == "kv_pool_tokens" and m["status"] == "measured"]
    assert len(pools) <= len(raw)


# ---- D9 -------------------------------------------------------------------------------------

def test_d9_lever_rows_without_a_key_do_not_crash_the_store(tmp_path):
    (tmp_path / "levers.jsonl").write_text('{"axis": "REL"}\n{"lever_id": "a"}\n')
    s = Store(tmp_path)
    assert len(s.tables["levers"]) == 2


def test_d9_missing_ingest_path_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["--db", str(tmp_path / "db"), "ingest", str(tmp_path / "nope")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_d9_unknown_goal_workload_is_a_clean_error(tmp_path, capsys):
    g = tmp_path / "g.toml"
    g.write_text(GOAL.replace("codex-8task-1agent", "no-such-workload"))
    assert cli.main(["--db", str(tmp_path / "db"), "report", str(g)]) == 2
    assert "unknown workload" in capsys.readouterr().err


def test_d9_two_agent_measure_dir_is_classified_as_two_agent(fx, tmp_path):
    dst = tmp_path / "measure"
    shutil.copytree(fx / "two-agent-hq-v3" / "measure", dst)
    (dst / "job-agent-0-x.json").write_text("{}")          # the real measure/ leg carries job-*.json next to SUMMARY.json
    assert classify_dir(dst) == "two-agent"


def test_d9_vram_device_index_fallback_is_stable_across_jobs(fx, db):
    d = fx / "campaign-llama-lost4"
    m = json.loads((d / "manifest.json").read_text())
    del m["devices"]
    (d / "manifest.json").write_text(json.dumps(m))
    Path(str(d / "manifest.json") + ".sha256").unlink()
    run([d], db=db)
    gpus = {m["dims"]["gpu"] for m in Store(db).rows("measurements") if m["metric"] == "vram_min_free_mib"}
    assert gpus == {0, 1}


def test_d9_two_agent_memory_csv_and_runaway_calls_are_used(fx, tmp_path):
    root = tmp_path / "two-agent-x"
    shutil.copytree(fx / "two-agent-hq-v3", root)
    (root / "memory.csv").write_text("index, uuid, memory.used [MiB], memory.total [MiB]\n0, GPU-a, 108 MiB, 16311 MiB\n1, GPU-b, 15 MiB, 16311 MiB\n")
    sp = root / "measure" / "SUMMARY.json"
    s = json.loads(sp.read_text())
    s["agents"][0]["runaway_calls"] = [{"output_tokens": 40000}]
    s["jobs"] = [{"task": "rust/forth", "runaway_calls": [{"output_tokens": 40000}]}, {"task": "rust/react", "runaway_calls": []}]
    sp.write_text(json.dumps(s))
    b = two_agent.parse(root)
    by = {(m["metric"], m["dims"].get("gpu")): m["value"] for m in b.measurements}
    assert by[("gpu_idle_used_mib", 0)] == 108.0 and by[("gpu_idle_used_mib", 1)] == 15.0
    assert by[("runaway_calls", None)] == 1 and by[("runaway_rate", None)] == 0.5
    s["agents"][0]["runaway_calls"] = 2          # a count instead of a list must not crash either
    sp.write_text(json.dumps(s))
    assert {m["value"] for m in two_agent.parse(root).measurements if m["metric"] == "runaway_calls"} == {2}


# ---- surviving mutants of the first review ---------------------------------------------------

def test_m7_queue_round_ms_matches_the_summary_md_it_replaces(fx, tmp_path):
    """round_ms = decode_s*1000/(n - accepted). queue/out/np2-fdown6mid/summary.md prints the same run (item 83) as
    8192 -> 39.88, 32768 -> 41.61, 98304 -> 46.59 round ms (q8-fix, np 1, n=1 boot)."""
    b = queue_np.parse(fx / "queue-np/raw/02-q8-fix-r1.json")
    got = {m["dims"]["depth"]: m["value"] for m in b.measurements if m["metric"] == "round_ms"}
    assert {d: round(v, 2) for d, v in got.items()} == {8192: 39.88, 32768: 41.61, 98304: 46.59}


def test_m9_timed_out_or_unverified_task_invalidates_the_campaign(fx):
    for key, val in (("timed_out", True), ("harness_verified", False)):
        d = fx / f"c-{key}"
        shutil.copytree(fx / "campaign-hq-lost4", d)
        job = sorted(d.glob("job-*.json"))[0]
        j = json.loads(job.read_text())
        j["probe"][key] = val
        job.write_text(json.dumps(j))
        Path(str(job) + ".sha256").unlink()
        b = campaign.parse(d)
        wall = next(m for m in b.measurements if m["metric"] == "wall_s")
        assert wall["status"] == "invalid" and "harness_verified false or timed_out" in wall["reason"]


def test_m14_control_kld_rows_never_set_a_candidates_kld(tmp_path):
    s = base_world(tmp_path)
    swift_llama(s, "J4-bf16", 100.0)
    from scripts.opt.ingest.kld import weights_config
    c = weights_config("swift-1.5:J4-bf16", "q8_0/q8_0")
    s.upsert("configs", c)
    for v, ctl in ((0.0030, False), (0.0001, True)):
        s.upsert("measurements", measurement(source="kld", metric="kld@REFBF", value=v, unit="nats", config_id=c["config_id"],
                                             workload_id="kld:x", prov=PROV, dims={"bucket": "all_prefill", "arm": "J4", **({"control": True} if ctl else {})},
                                             item_id=str(v), ident=str(v)))
    assert report.kld_table(s)[("swift-1.5:J4-bf16", "kld@REFBF")]["q8_0/q8_0"] == (0.0030, 1)


def test_m4_kld_prefers_the_exact_kv_when_several_exist(tmp_path):
    s = base_world(tmp_path)
    swift_llama(s, "J4-bf16", 100.0, kv="q8_0")
    from scripts.opt.ingest.kld import weights_config
    for kv, v in (("f16/f16", 0.0010), ("q8_0/q8_0", 0.0020), ("f16/q8_0", 0.0030)):
        c = weights_config("swift-1.5:J4-bf16", kv)
        s.upsert("configs", c)
        s.upsert("measurements", measurement(source="kld", metric="kld@REFBF", value=v, unit="nats", config_id=c["config_id"],
                                             workload_id="kld:x", prov=PROV, dims={"bucket": "all_prefill", "arm": kv},
                                             item_id=kv, ident=kv))
    _, ev = render(s)
    assert next(r for r in ev["rows"] if r["cand"]["engine"] == "llama")["kld"]["kld@REFBF"][0] == 0.0020


def test_m16_default_rho_is_zero_in_wall_est_two():
    a = [[1000, 1000, 100, 0, 0.0, 1.0]] * 2
    b = [[1000, 500, 400, 0, 0.0, 1.0]] * 2
    prof = {"fixed_call_s": {"hq": 0.0}, "runs": [{"agents": {"0": a, "1": b}}]}
    assert W.wall_est_two(prof, 1000.0, 100.0, "hq")[0] == W.wall_est_two(prof, 1000.0, 100.0, "hq", rho=0.0)[0]
    assert W.wall_est_two(prof, 1000.0, 100.0, "hq")[0] != W.wall_est_two(prof, 1000.0, 100.0, "hq", rho=1.0)[0]


def test_m15_kld_ids_follow_content_not_directory_name(tmp_path):
    p1, p2 = tmp_path / "x" / "out-b1" / "results.txt", tmp_path / "y" / "out-b1" / "results.txt"
    for p, v in ((p1, "0.00232"), (p2, "0.00999")):
        p.parent.mkdir(parents=True)
        p.write_text(KLD_TXT.format(kld=v))
    ids = [{m["id"] for m in kld.parse(p).measurements} for p in (p1, p2)]
    assert not (ids[0] & ids[1])
    again = {m["id"] for m in kld.parse(p1).measurements}
    assert again == ids[0]


def test_d3_foreign_plan_md_is_skipped_not_rejected(tmp_path):
    from scripts.opt.ingest import ledger
    from scripts.opt.store import Skip
    plan = tmp_path / "PLAN.md"
    plan.write_text("# some other plan\nnothing here\n")
    with pytest.raises(Skip):
        ledger.parse(plan)


def _row(klass, verdict, unver, wall, pct=10.0):
    return {"cand": {"klass": klass}, "verdict": verdict, "kld_unverified": unver, "wall_est": wall, "tie_pct": pct}


def test_d4_tie_bands_never_span_basis_classes_verdicts_or_kld_verification():
    rows = [_row("measured", "PASS", False, 100.0), _row("F1", "PASS", False, 200.0),
            _row("F1", "UNKNOWN", False, 300.0), _row("F1", "UNKNOWN", True, 400.0)]
    report._tie_bands(rows)
    assert [r["band"] for r in rows] == [1, 1, 1, 1]   # each starts its own band 1; merging any two would number a later one 2


def test_d4_tie_band_is_anchored_on_the_bands_best_estimate_not_chained():
    rows = [_row("measured", "PASS", False, w) for w in (100.0, 108.0, 116.0, 125.0)]
    report._tie_bands(rows)
    assert [r["band"] for r in rows] == [1, 1, 2, 2]


def test_d8_derived_vram_uses_the_loo_band_in_the_report(tmp_path):
    s = base_world(tmp_path)
    cfg = add_cfg(s, llama_cmd(SWIFT + "NOVRAM-bf16", 1, 102400, "q8_0"), stack="s")
    for metric, v in (("decode_tok_s", 100.0), ("prefill_tok_s", 800.0), ("output_tokens", 100000.0), ("prefill_new_tokens", 100000.0)):
        add_m(s, cfg, metric, v, item="nv")
    with_kld(s, ["NOVRAM-bf16"])
    from scripts.opt import fit as fitmod
    real = fitmod.predict_free
    fitmod.predict_free = lambda model, cfg, **k: (700.0, "stub")
    band = fitmod.loo_band(fitmod.fit_llama(s))
    try:
        _, ev = render(s)
    finally:
        fitmod.predict_free = real
    r = next(r for r in ev["rows"] if r["cand"]["engine"] == "llama")
    assert r["vram"] == ("derived", 700.0) and r["vram_class"] == fitmod.classify(700.0, band) == "probe-needed" and fitmod.classify(700.0, 0.0) == "fits"


def test_d8_classify_reject_also_respects_the_band():
    assert fit.classify(fit.REJECT_MIB - 10) == "reject"
    assert fit.classify(fit.REJECT_MIB - 10, band=50) == "probe-needed"


def test_d8_loo_band_has_a_floor():
    assert fit.loo_band({"loo": [{"resid": 3}, {"resid": -2}, {"resid": 1}]}) == fit.RESID_FLOOR_MIB


def test_d1_vram_groups_do_not_pool_model_versions_sharing_a_label():
    mk = lambda mid: {"engine": "llama", "model_id": mid, "model_label": "IQ4_XS", "kv_k": "q8_0", "kv_v": "q8_0", "k": 3, "ub": 128}
    assert fit.group_key(mk("swift-1.5:IQ4_XS")) != fit.group_key(mk("swift-1.0:IQ4_XS"))


def test_m4_queue_streams_without_a_server_clock_are_skipped_not_crashed(fx, tmp_path):
    src = json.loads((fx / "queue-np/raw/02-q8-fix-r1.json").read_text())
    for res in src["results"]:
        for rnd in res["rounds"]:
            extra = {k: v for k, v in rnd["streams"][0].items() if k != "server_tg"}
            rnd["streams"].append({**extra, "draft_accepted": 0})
    p = tmp_path / "q" / "02-q8-fix-r1.json"
    p.parent.mkdir()
    p.write_text(json.dumps(src))
    (p.with_suffix(".launch.json")).write_text((fx / "queue-np/raw/02-q8-fix-r1.launch.json").read_text())
    got = {m["dims"]["depth"]: round(m["value"], 2) for m in queue_np.parse(p).measurements if m["metric"] == "round_ms"}
    assert got == {8192: 39.88, 32768: 41.61, 98304: 46.59}


def test_m14_arm_level_control_rows_are_flagged_control_on_ingest(tmp_path):
    s = ingest_kld(tmp_path, {"J5q_f16_off": 0.0024, "J5q_f16_off_rep": 0.0001})
    ctl = {m["dims"]["arm"]: m["dims"].get("control", False) for m in s.rows("measurements")}
    assert ctl == {"J5q_f16_off": False, "J5q_f16_off_rep": True}


def test_d6_a_not_run_task_adds_no_verified_wall_and_does_not_fake_a_wall_mismatch(fx, db):
    d = fx / "campaign-llama-lost4"
    job = sorted(d.glob("job-*.json"))[0]
    task = json.loads(job.read_text())["task"]
    job.write_text(json.dumps({"status": "failed", "model": "m", "task": task, "attempt": "attempts/x", "reason": "r", "finished_at": 1.0}))
    Path(str(job) + ".sha256").unlink()
    run([d], db=db)
    reasons = " ".join(m["reason"] or "" for m in Store(db).rows("measurements") if m["metric"] == "wall_s")
    assert "without verified_wall_seconds" not in reasons and "did not run" in reasons


def test_d4_held_out_tiers_never_see_the_held_out_configs_calls_or_constants(tmp_path, monkeypatch):
    s = Store(tmp_path)
    two_config_campaigns(s, [(3, 1000.0, 100.0), (3, 1000.0, 200.0), (4, 1000.0, 100.0), (5, 1000.0, 100.0), (6, 1000.0, 100.0)])
    n_cfg = len({w["config_id"] for w in W._valid_calls(s, "campaign-calls")})
    profiles = W.build_profiles(s, write=False)
    pools, fits = [], []
    real_pool, real_fit = W._pool, W.fit_constants
    monkeypatch.setattr(W, "_pool", lambda prof, camps: (pools.append({c["config_id"] for c in camps}), real_pool(prof, camps))[1])
    monkeypatch.setattr(W, "fit_constants", lambda camps, *a: (fits.append({c["config_id"] for c in camps}), real_fit(camps, *a))[1])
    res = W.calibrate(s, profiles)
    assert res["rows"]["B"] and pools and fits
    assert all(len(p) == n_cfg - 1 for p in pools) and all(len(f) == n_cfg - 1 for f in fits)


def test_d7_a_row_differing_only_in_campaign_name_or_path_is_the_same_row(tmp_path):
    s = Store(tmp_path)
    row = {"id": "calls:abc", "kind": "campaign-calls", "workload_id": "w", "config_id": "c", "tasks": {"t": [[1, 2]]},
           "campaign": "dirA@abc", "out_dir": "/a/dirA", "summary_sha256": "abc"}
    assert s.upsert("workloads", row) == "added"
    s.upsert("workloads", {**row, "campaign": "dirB@abc", "out_dir": "/b/dirB"})
    kept = s.tables["workloads"]["calls:abc"]
    assert kept["tasks"] == row["tasks"] and "/b/dirB" in json.dumps(kept.get("also_at"))   # merged as a copy, not replaced
