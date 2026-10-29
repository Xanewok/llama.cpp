import json
import shutil
from pathlib import Path

import pytest

from scripts.opt.ingest import campaign, classify_dir, classify_file, discover, graph_reuse, hq_conc, kld, queue_np, run, two_agent, weekend
from scripts.opt.ingest.kld import arm_identity
from scripts.opt.store import Reject, Skip, Store


def by_metric(batch, metric, **dims):
    return [m for m in batch.measurements if m["metric"] == metric and all(m["dims"].get(k) == v for k, v in dims.items())]


# ---- campaign ---------------------------------------------------------------------------------

def test_hq_campaign_row_values_match_the_summary(fx):
    d = fx / "campaign-hq-lost4"
    b = campaign.parse(d)
    summ = json.loads((d / "SUMMARY.json").read_text())
    (wall,) = by_metric(b, "wall_s")
    assert wall["value"] == summ["wall_seconds"] and wall["status"] == "measured" and wall["unit"] == "s"
    (solve,) = by_metric(b, "solve_rate")
    assert solve["value"] == summ["solved"] / summ["tasks"]
    assert len(by_metric(b, "task_wall_s")) == summ["tasks"]
    (cfg,) = b.configs.values()
    assert cfg["engine"] == "hq" and cfg["model_label"].startswith("hq:") and cfg["np"] >= 1
    calls = next(w for w in b.workloads.values() if w["kind"] == "campaign-calls")
    assert calls["valid"] and sum(len(v) for v in calls["tasks"].values()) == by_metric(b, "model_calls")[0]["value"]
    assert all(len(c) == 6 and c[0] >= c[1] >= 0 for t in calls["tasks"].values() for c in t)


def test_llama_campaign_has_config_provenance_and_no_call_table(fx):
    b = campaign.parse(fx / "campaign-llama-lost4")
    (cfg,) = b.configs.values()
    assert cfg["engine"] == "llama" and cfg["model_sha256"] and cfg["build_sha"] and cfg["ctx"]
    assert not [w for w in b.workloads.values() if w["kind"] == "campaign-calls"]
    assert by_metric(b, "vram_min_free_mib")


def test_campaign_rejects_corrupt_job_json(fx):
    d = fx / "campaign-hq-lost4"
    job = sorted(d.glob("job-*.json"))[0]
    job.write_text(job.read_text()[:200])
    with pytest.raises(Reject, match="sha256 sidecar mismatch"):
        campaign.parse(d)
    Path(str(job) + ".sha256").unlink()
    with pytest.raises(Reject, match="corrupt JSON"):
        campaign.parse(d)


def test_campaign_rejects_sha_sidecar_mismatch(fx):
    d = fx / "campaign-hq-lost4"
    job = sorted(d.glob("job-*.json"))[0]
    j = json.loads(job.read_text())
    j["status"] = "failed" if j["status"] == "passed" else "passed"
    job.write_text(json.dumps(j))
    with pytest.raises(Reject, match="sha256 sidecar mismatch"):
        campaign.parse(d)


def test_campaign_rejects_passed_job_without_probe(fx):
    d = fx / "campaign-hq-lost4"
    job = sorted(d.glob("job-*.json"))[0]
    j = json.loads(job.read_text())
    del j["probe"]
    job.write_text(json.dumps(j))
    Path(str(job) + ".sha256").unlink()
    with pytest.raises(Reject, match="passed job without a probe"):
        campaign.parse(d)


def test_campaign_wall_sum_mismatch_is_marked_invalid_not_ingested_as_valid(fx):
    d = fx / "campaign-hq-lost4"
    s = json.loads((d / "SUMMARY.json").read_text())
    s["wall_seconds"] *= 2
    (d / "SUMMARY.json").write_text(json.dumps(s))
    b = campaign.parse(d)
    assert {m["status"] for m in b.measurements} == {"invalid"} or all(
        m["status"] == "invalid" for m in b.measurements if m["metric"] == "wall_s")
    assert "sum(task wall)" in by_metric(b, "wall_s")[0]["reason"]
    assert not next(w for w in b.workloads.values() if w["kind"] == "campaign-calls")["valid"]
    assert b.flagged


def test_campaign_rejects_truncated_model_calls(fx):
    d = fx / "campaign-hq-lost4"
    mc = sorted(d.glob("attempts/*/model-calls.jsonl"))[0]
    mc.write_text(mc.read_text()[:-40])
    with pytest.raises(Reject, match="corrupt model-calls"):
        campaign.parse(d)


def test_campaign_without_summary_or_jobs_is_rejected(fx):
    d = fx / "campaign-hq-lost4"
    (d / "SUMMARY.json").unlink()
    with pytest.raises(Reject, match="no SUMMARY.json"):
        campaign.parse(d)
    d2 = fx / "campaign-llama-lost4"
    for f in list(d2.glob("job-*")):
        f.unlink()
    with pytest.raises(Reject, match="no job"):
        campaign.parse(d2)


# ---- weekend ----------------------------------------------------------------------------------

def test_weekend_summary_with_and_without_campaign_dir(fx):
    b = weekend.parse(fx / "weekend/summary.json")
    assert {m["metric"] for m in b.measurements} >= {"wall_s", "round_ms", "weekend_eligible"}
    assert len(b.configs) == 2
    assert any("campaign dir missing" in m for _, m in b.flagged)
    rows = json.loads((fx / "weekend/summary.json").read_text())
    assert rows[0]["round_ms"] in [m["value"] for m in by_metric(b, "round_ms")]


def test_weekend_ineligible_row_is_invalid(fx):
    p = fx / "weekend/summary.json"
    rows = json.loads(p.read_text())
    rows[1]["eligible"] = False
    p.write_text(json.dumps(rows))
    b = weekend.parse(p)
    ineligible = [m for m in b.measurements if m["item_id"].startswith("phase") and "missing" not in m["item_id"] and m["status"] == "invalid"]
    assert ineligible and {m["reason"] for m in ineligible} == {"weekend summary eligible=false"}
    assert all(m["status"] == "measured" for m in b.measurements if m["metric"] == "wall_s" and m["item_id"] == Path(rows[0]["path"]).name)


def test_weekend_rejects_non_list_and_missing_columns(fx):
    p = fx / "weekend/summary.json"
    p.write_text('{"a": 1}')
    with pytest.raises(Reject):
        weekend.parse(p)
    p.write_text(json.dumps([{"phase": 1}]))
    with pytest.raises(Reject, match="no row could be ingested"):
        weekend.parse(p)


# ---- graph-reuse ------------------------------------------------------------------------------

def test_graph_reuse_status_follows_the_producer_suffix(fx):
    b = graph_reuse.parse(fx / "graph-reuse-ab/ab-summary.json")
    st = {m["metric"]: m["status"] for m in b.measurements}
    assert st["prompt_ms"] == "measured" and st["steady_round_ms"] == "derived" and st["delta_request_wall_ms"] == "derived"
    assert len(b.configs) >= 2


def test_graph_reuse_requires_command_json_and_numeric_values(fx):
    g = fx / "graph-reuse-ab"
    victim = next(p for p in g.iterdir() if p.is_dir())
    (victim / "command.json").unlink()
    with pytest.raises(Reject, match="command.json missing"):
        graph_reuse.parse(g / "ab-summary.json")


def test_graph_reuse_rejects_non_numeric_cell(fx):
    p = fx / "graph-reuse-ab/ab-summary.json"
    d = json.loads(p.read_text())
    blk = d if isinstance(d, dict) else d[0]
    blk["requests"][0]["prompt_ms_M"] = "n/a"
    p.write_text(json.dumps(d))
    with pytest.raises(Reject, match="non-numeric"):
        graph_reuse.parse(p)


# ---- hq-conc ----------------------------------------------------------------------------------

def test_hq_conc_schema1_boot_is_ingested(fx):
    b = hq_conc.parse(fx / "hq-conc/hq-2x94k/raw/boot1")
    assert by_metric(b, "kv_pool_tokens") and by_metric(b, "decode_tok_s", depth=94000)
    (cfg,) = b.configs.values()
    assert cfg["engine"] == "hq" and cfg["k"] and cfg["gpu_util"]
    assert {m["metric"] for m in b.measurements} & {"agg_tok_s", "prefix_kept"}


def test_hq_prefix_cells_and_canary_control(fx):
    d = fx / "hq-conc/hq-prefix-loss-v3/measure/raw/boot1"
    b = hq_conc.parse(d)
    frac = by_metric(b, "prefix_kept_frac")
    assert frac and all(m["status"] == "measured" for m in frac)
    assert {m["value"] for m in frac} <= {0.0, 0.5, 1.0}
    pj = json.loads((d / "prefix.json").read_text())
    pj["canary"] = True
    (d / "prefix.json").write_text(json.dumps(pj))
    bad = hq_conc.parse(d)
    assert {m["status"] for m in by_metric(bad, "prefix_kept_frac")} == {"invalid"}
    assert by_metric(bad, "prefix_kept_frac")[0]["reason"] == "canary run"


def test_hq_conc_rejects_missing_fit_and_unparsable_serve_log(fx):
    d = fx / "hq-conc/hq-2x94k/raw/boot1"
    (d / "serve.log").write_text("nothing useful\n")
    with pytest.raises(Reject, match="non-default args"):
        hq_conc.parse(d)
    (d / "fit.json").unlink()
    with pytest.raises(Reject, match="no fit.json"):
        hq_conc.parse(d)


# ---- two-agent --------------------------------------------------------------------------------

def test_two_agent_run_and_call_tables(fx):
    b = two_agent.parse(fx / "two-agent-hq-v3")
    s = json.loads((fx / "two-agent-hq-v3/measure/SUMMARY.json").read_text())
    (wall,) = by_metric(b, "wall_s")
    assert wall["value"] == s["wall_seconds"] and wall["status"] == "measured"
    w = next(w for w in b.workloads.values() if w["kind"] == "two-agent-calls")
    assert set(w["agents"]) == {"0", "1"} and all(w["agents"][a] for a in w["agents"])
    assert b.workloads["two-agent-8task"]["agents"] == 2


def test_two_agent_canary_only_dir_and_failed_verdict(fx):
    d = fx / "two-agent-hq-v3"
    s = json.loads((d / "measure/SUMMARY.json").read_text())
    s["valid"], s["verdict"] = False, "INVALID_PREFIX_LOST"
    (d / "measure/SUMMARY.json").write_text(json.dumps(s))
    b = two_agent.parse(d)
    assert {m["status"] for m in by_metric(b, "wall_s")} == {"invalid"}
    (d / "measure/SUMMARY.json").unlink()
    with pytest.raises(Reject, match="canary-only"):
        two_agent.parse(d)


# ---- queue np ---------------------------------------------------------------------------------

def test_queue_np_round_ms_and_vram(fx):
    run_json = next(p for p in (fx / "queue-np/raw").glob("*.json") if not p.name.endswith(".launch.json"))
    b = queue_np.parse(run_json)
    assert by_metric(b, "round_ms") and by_metric(b, "vram_min_free_mib")
    smi = run_json.with_suffix(".smi.csv").read_text().splitlines()
    want = {}
    for line in smi:
        r = line.split(",")
        want[int(r[1])] = min(want.get(int(r[1]), 1e18), float(r[3]))
    got = {m["dims"]["gpu"]: m["value"] for m in by_metric(b, "vram_min_free_mib")}
    assert got == want


def test_queue_np_negative_controls(fx):
    run_json = next(p for p in (fx / "queue-np/raw").glob("*.json") if not p.name.endswith(".launch.json"))
    smi = run_json.with_suffix(".smi.csv")
    smi.write_text("garbage\n")
    with pytest.raises(Reject, match="malformed|non-numeric"):
        queue_np.parse(run_json)
    smi.unlink()
    assert any("no smi.csv" in m for _, m in queue_np.parse(run_json).flagged)
    d = json.loads(run_json.read_text())
    d["np"] += 1
    run_json.write_text(json.dumps(d))
    with pytest.raises(Reject, match="disagree with launch record"):
        queue_np.parse(run_json)
    with pytest.raises(Reject, match="launch record"):
        queue_np.parse(run_json.with_suffix(".launch.json"))


# ---- KLD --------------------------------------------------------------------------------------

def test_kld_refs_are_part_of_the_metric_and_never_mixed():
    b = kld.parse(Path(__file__).parent / "fixtures/kld/kld-j5-b1/results.txt")
    refs = {m["metric"].split("@")[1] for m in b.measurements if m["metric"].startswith("kld@")}
    assert refs <= {"REF6", "REF8", "REFBF"} and refs
    assert all("@" in m["metric"] for m in b.measurements if m["metric"].startswith("kld"))


def test_kld_refbf_json_and_vast_tables(fx):
    j = kld.parse(fx / "kld/vast-refbf/results.json")
    assert {m["metric"] for m in j.measurements} <= {"kld@REF6", "kld@REF8", "kld@REFBF"} and "kld@REFBF" in {m["metric"] for m in j.measurements}
    t = kld.parse(fx / "kld/vast-kld-l/results.txt")
    assert {m["metric"] for m in t.measurements} == {"kld@REF6"}
    assert {m["dims"]["bucket"] for m in t.measurements} >= {"all_prefill", "0-8k"}


def test_kld_table_without_reference_is_skipped_unknown_reference_rejected(tmp_path):
    p = tmp_path / "results.txt"
    p.write_text("arm  kld  se\nJ4  0.0033  0.0003\n")
    with pytest.raises(Skip):
        kld.parse(p)
    p.write_text("== KL(REFXX || arm), 3 seqs; n_seq = 3; ...\narm +GB MiB/GPU KLD seqSE\nJ4 0.0 +0 0.0033 0.0003 | control\n")
    with pytest.raises(Reject, match="unknown reference"):
        kld.parse(p)
    j = tmp_path / "results.json"
    j.write_text(json.dumps({"refs": {}, "kld": {"J4": {"REFZZ": {"all_prefill": [0.003, 0.0003, [1]]}}}}))
    with pytest.raises(Reject, match="unknown reference"):
        kld.parse(j)
    j.write_text(json.dumps({"refs": {}, "kld": {"J4": {"REFBF": {"all_prefill": [-1.0, 0.0003, [1]]}}}}))
    with pytest.raises(Reject, match="bad KLD cell"):
        kld.parse(j)


@pytest.mark.parametrize("arm,want", [
    ("J5q", ("swift-1.5:J5q-bf16", "f16/f16")),
    ("loc_j4q8", ("swift-1.5:J4-fromQ8", "f16/f16")),
    ("loc_j4bf16", ("swift-1.5:J4-bf16", "f16/f16")),
    ("fdown6mid", ("swift-1.5:fdown6mid-bf16", "f16/f16")),
    ("IQ4_XS, KV K f16 / V q8_0", ("swift-1.5:IQ4_XS", "f16/q8_0")),
    ("Q6_K, KV q8_0/q8_0", ("swift-1.5:Q6_K", "q8_0/q8_0")),
    ("l40_J1_f16", ("swift-1.5:IQ4_XS-ssmout6", "f16/f16")),
    ("J1 (IQ4_XS + ssm_out Q6_K), KV f16", ("swift-1.5:IQ4_XS-ssmout6", "f16/f16")),
    ("A0", (None, "unknown")),
    ("J5q_q8", (None, "unknown")),
])
def test_arm_identity(arm, want):
    assert arm_identity(arm)[:2] == want


# ---- driver -----------------------------------------------------------------------------------

def test_ingest_all_fixtures_is_idempotent_and_bytewise_stable(fx, db):
    first = run([fx], db=db)
    assert not first["rejected"] and set(first["counts"]) == {"campaign", "graph-reuse-ab", "hq-conc", "kld", "queue-np", "two-agent", "weekend"}
    snap = {p.name: p.read_bytes() for p in db.glob("*.jsonl")}
    second = run([fx], db=db)
    assert all(u["added"] == 0 and u["updated"] == 0 for u in second["units"])
    assert {p.name: p.read_bytes() for p in db.glob("*.jsonl")} == snap


def test_ingest_dry_run_writes_nothing(fx, db):
    run([fx], db=db, dry_run=True)
    assert not db.exists() or not list(db.glob("*.jsonl"))


def test_rejects_do_not_reach_the_db(fx, db):
    d = fx / "campaign-hq-lost4"
    job = sorted(d.glob("job-*.json"))[0]
    job.write_text("{")
    rep = run([d], db=db)
    assert rep["rejected"] and not rep["units"]
    assert not list(db.glob("measurements.jsonl")) or not Store(db).rows("measurements")


def test_unrecognised_path_is_reported(tmp_path, db):
    (tmp_path / "x.json").write_text("{}")
    rep = run([tmp_path], db=db)
    assert rep["rejected"] and "no recognisable source" in rep["rejected"][0][1]


def test_classification_is_structural(fx, tmp_path):
    assert classify_file(fx / "weekend/summary.json") == "weekend"
    other = tmp_path / "summary.json"
    other.write_text("[]")
    assert classify_file(other) is None
    other.write_text("{broken")
    assert classify_file(other) is None
    assert classify_dir(fx / "campaign-hq-lost4") == "campaign"
    assert classify_dir(fx / "two-agent-hq-v3") == "two-agent"
    assert classify_dir(fx / "hq-conc/hq-2x94k/raw/boot1") == "hq-conc"
    kinds = sorted(k for k, _ in discover(fx))
    assert kinds.count("campaign") >= 3 and "queue-np" in kinds and "graph-reuse-ab" in kinds
