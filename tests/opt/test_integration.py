"""Real files in place; skipped where the sources are absent (other checkouts, scratch dirs)."""
from pathlib import Path

import pytest

from scripts.opt import fit, workload as W
from scripts.opt.ingest import run
from scripts.opt.store import ROOT, Store

RESULTS = ROOT / "results"
pytestmark = pytest.mark.skipif(not RESULTS.exists(), reason="repo results tree not present")


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    db = tmp_path_factory.mktemp("db")
    rep = run([RESULTS], db=db)
    return db, rep


def test_repo_results_tree_ingests_with_rejects_listed_not_hidden(ingested):
    db, rep = ingested
    assert rep["counts"]["campaign"]["units"] >= 20 and rep["counts"]["queue-np"]["units"] >= 10
    assert all(why for _, why in rep["rejected"])
    st = Store(db)
    assert {m["status"] for m in st.rows("measurements")} <= {"measured", "derived", "ledger-prior", "invalid"}
    assert all(m["provenance"].get("summary_sha256") for m in st.rows("measurements"))


def test_repo_reingest_is_a_noop(ingested):
    db, _ = ingested
    before = {p.name: p.read_bytes() for p in db.glob("*.jsonl")}
    again = run([RESULTS], db=db)
    assert all(u["added"] == 0 and u["updated"] == 0 for u in again["units"])
    assert {p.name: p.read_bytes() for p in db.glob("*.jsonl")} == before


def test_no_two_references_share_a_metric(ingested):
    db, _ = ingested
    for m in Store(db).rows("measurements"):
        if m["source"] == "kld":
            assert m["metric"].startswith(("kld@", "kld_delta@"))


def test_vram_fit_has_a_small_median_residual_on_real_boots(ingested):
    db, _ = ingested
    m = fit.fit_llama(Store(db))
    loo = sorted(abs(p["resid"]) for p in m["loo"])
    if not m["points"]:
        pytest.skip("no group with two contexts in the repo tree alone (the np1/np2 boots live in scratch dirs)")
    assert len(loo) > 20 and loo[len(loo) // 2] < 150


def test_wall_est_own_calls_error_on_real_campaigns(ingested):
    db, _ = ingested
    res = W.calibrate(Store(db))
    a = next(v for k, v in res["tiers"].items() if k.startswith("A "))
    assert a["n"] >= 5 and a["median_abs_pct"] < 5


EXTERNAL = {
    "weekend": Path.home() / "repos/qwen38-weekend/results",
    "graph-reuse-ab": Path.home() / "repos/graph-reuse/codex/runs",
    "hq-conc": Path.home() / "repos/hq-conc/queue-out",
    "two-agent": Path.home() / "repos/two-agent/results/two-agent-hq-v3",
}


@pytest.mark.parametrize("kind", sorted(EXTERNAL))
def test_external_checkouts_in_place(kind, tmp_path):
    path = EXTERNAL[kind]
    if not path.exists():
        pytest.skip(f"{path} not present")
    rep = run([path], db=tmp_path)
    assert rep["counts"].get(kind, {}).get("units", 0) >= 1
    assert rep["counts"][kind]["measurements"] > 0
