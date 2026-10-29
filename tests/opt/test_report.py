import pytest

from scripts.opt import goal as goalmod, report
from scripts.opt.ingest.base import measurement
from scripts.opt.store import Store

from .synth import PROV, add_cfg, add_m, llama_cmd
from .test_workload import PROFILE, synthetic_campaigns

GOAL = '''[goal]
name = "t"
workload = "codex-8task-1agent"
primary = "wall_est"
tiebreak = ["kld@REFBF", "vram_min_free_mib"]
[constraints]
fits = { np = 1, ctx_per_slot = 98304 }
kld_REFBF = { max = 0.0035 }
vram_min_free_mib = { min = 300 }
'''


def kld_row(store, quant, value, ref="REFBF", kv="q8_0"):
    from scripts.opt.ingest.kld import weights_config
    c = weights_config(f"other:{quant}", f"{kv}/{kv}")
    store.upsert("configs", c)
    store.upsert("measurements", measurement(source="kld", metric=f"kld@{ref}", value=value, unit="nats", config_id=c["config_id"],
                                             workload_id="kld:x", prov=PROV, dims={"bucket": "all_prefill", "arm": quant},
                                             item_id=quant + ref, ident=quant + ref))


def llama_candidate(store, quant, decode, prefill=800.0, free=2000.0, **kw):
    cfg = add_cfg(store, llama_cmd(quant, **kw), stack=f"stack-{quant}")
    add_m(store, cfg, "decode_tok_s", decode, item=quant)
    add_m(store, cfg, "prefill_tok_s", prefill, item=quant)
    add_m(store, cfg, "vram_min_free_mib", free, gpu=0, item=quant)
    add_m(store, cfg, "output_tokens", 100000.0, item=quant)
    add_m(store, cfg, "prefill_new_tokens", 100000.0, item=quant)
    return cfg


def world(tmp_path):
    s = Store(tmp_path)
    synthetic_campaigns(s, n=3)
    s.upsert("workloads", {**PROFILE, "id": "profile:codex-8task-1agent", "kind": "profile", "n_campaigns": 1,
                           "sources": ["x"], "workload_id": "codex-8task-1agent"})
    llama_candidate(s, "FAST", 120.0)
    llama_candidate(s, "MID", 90.0)
    llama_candidate(s, "SLOW", 60.0)
    llama_candidate(s, "NOKLD", 150.0)
    llama_candidate(s, "TIGHT", 130.0, free=100.0)
    llama_candidate(s, "NP2", 200.0, np_=2, ctx=100000)
    kld_row(s, "FAST", 0.0040)
    kld_row(s, "MID", 0.0030)
    kld_row(s, "SLOW", 0.0020)
    kld_row(s, "TIGHT", 0.0030)
    kld_row(s, "NP2", 0.0030)
    kld_row(s, "MID", 0.0001, ref="REF6")
    return s


def render(s, text=GOAL):
    return report.render(s, goalmod.parse(text), show_fail=True)


def test_constraints_decide_the_verdict(tmp_path):
    _, ev = render(world(tmp_path))
    v = {r["cand"]["model_id"].split(":", 1)[-1]: r for r in ev["rows"] if r["cand"]["engine"] == "llama"}
    assert v["MID"]["verdict"] == "PASS" and v["SLOW"]["verdict"] == "PASS"
    assert v["FAST"]["verdict"] == "FAIL" and v["FAST"]["checks"]["kld@REFBF"] is False
    assert v["NOKLD"]["verdict"] == "UNKNOWN" and v["NOKLD"]["checks"]["kld@REFBF"] is None
    assert v["TIGHT"]["verdict"] == "FAIL" and v["TIGHT"]["checks"]["vram_min_free_mib"] is False
    assert v["NP2"]["verdict"] == "FAIL" and v["NP2"]["checks"]["fits"] is False  # 100000 / 2 slots < 98304


def test_ranking_is_by_wall_est_within_verdict_and_unknown_never_outranks_pass(tmp_path):
    _, ev = render(world(tmp_path))
    order = [r["verdict"] for r in ev["rows"]]
    assert order == sorted(order, key={"PASS": 0, "UNKNOWN": 1, "FAIL": 2}.get)
    passing = [r for r in ev["rows"] if r["verdict"] == "PASS"]
    assert [r["cand"]["model_id"].split(":", 1)[-1] for r in passing] == ["MID", "SLOW"]
    assert passing[0]["wall_est"] < passing[1]["wall_est"]


def test_reference_is_never_mixed(tmp_path):
    s = world(tmp_path)
    _, ev = render(s)
    mid = next(r for r in ev["rows"] if r["cand"]["model_id"].split(":", 1)[-1] == "MID")
    assert mid["kld"]["kld@REFBF"][0] == pytest.approx(0.0030)
    _, ev6 = render(s, GOAL.replace("kld_REFBF", "kld_REF6").replace('"kld@REFBF"', '"kld@REF6"'))
    mid6 = next(r for r in ev6["rows"] if r["cand"]["model_id"].split(":", 1)[-1] == "MID")
    assert mid6["kld"]["kld@REF6"][0] == pytest.approx(0.0001)
    slow6 = next(r for r in ev6["rows"] if r["cand"]["model_id"].split(":", 1)[-1] == "SLOW")
    assert slow6["kld"]["kld@REF6"] is None  # SLOW has only a REFBF value; it is not borrowed


def test_pareto_front_is_nondominated(tmp_path):
    text, ev = render(world(tmp_path))
    front = report.pareto(ev["rows"], "kld@REFBF")
    labels = [r["cand"]["model_id"].split(":", 1)[-1] for r in front]
    assert "SLOW" in labels and "FAST" in labels  # FAST fails the KLD bound but is the speed end of the front
    assert "NOKLD" not in labels and "TIGHT" not in labels  # no KLD / violates VRAM
    pts = [(r["wall_est"], r["metrics"]["kld@REFBF"]) for r in front]
    assert pts == sorted(pts) and all(a[1] > b[1] for a, b in zip(pts, pts[1:]))
    assert "## Pareto front" in text and "missing" in text


def test_report_renders_status_tags_and_unverified_column(tmp_path):
    text, _ = render(world(tmp_path))
    assert "unverified" in text and "d |" in text and " m " in text
    assert "kld@REFBF" in text and "Ranked" in text


def test_profile_without_sources_is_an_error(tmp_path):
    s = Store(tmp_path)
    llama_candidate(s, "A", 100.0)
    with pytest.raises((ValueError, StopIteration)):
        render(s)
