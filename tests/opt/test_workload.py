import pytest

from scripts.opt import workload as W
from scripts.opt.ingest import run
from scripts.opt.store import Store

from .synth import add_cfg, add_m, llama_cmd

PROFILE = {"id": "profile:t", "n_campaigns": 1, "tasks_per_campaign": 2, "task_overhead_s": 3.0, "fixed_call_s": {"hq": 0.1},
           "calls": [[10000, 1000, 100, 1, 2.0, 5.0], [20000, 500, 200, 1, 1.0, 6.0]]}


def test_tps_at_flat_curve_and_clamp():
    assert W.tps_at(50, 123) == 50.0
    curve = [(8000, 100.0), (32000, 60.0)]
    assert W.tps_at(curve, 1000) == 100.0 and W.tps_at(curve, 99999) == 60.0
    assert W.tps_at(curve, 20000) == pytest.approx(80.0)


def test_wall_est_single_hand_computed():
    est, parts = W.wall_est_single(PROFILE, 1000.0, 100.0, "hq")
    assert parts["fixed"] == pytest.approx(0.2) and parts["prefill"] == pytest.approx(1.5)
    assert parts["decode"] == pytest.approx(3.0) and parts["gap"] == pytest.approx(3.0) and parts["overhead"] == pytest.approx(6.0)
    assert est == pytest.approx(13.7)


def test_pooled_profile_is_the_mean_campaign():
    pooled = {**PROFILE, "n_campaigns": 2, "calls": PROFILE["calls"] * 2}
    assert W.wall_est_single(pooled, 1000.0, 100.0, "hq")[0] == pytest.approx(13.7)


def test_lost_prefix_costs_a_full_reprefill():
    est, parts = W.wall_est_single(PROFILE, 1000.0, 100.0, "hq", kept_fn=lambda d: d <= 10000)
    assert parts["prefill"] == pytest.approx(1.0 + 20.0)
    assert est == pytest.approx(13.7 - 1.5 + 21.0)


def test_volume_rebase_scales_only_tokens():
    out_base = W.wall_est_single(PROFILE, 1000.0, 100.0, "hq")[1]
    out_dbl = W.wall_est_single(PROFILE, 1000.0, 100.0, "hq", volume={"out": 600, "new": 1500})[1]
    assert out_dbl["decode"] == pytest.approx(2 * out_base["decode"]) and out_dbl["prefill"] == pytest.approx(out_base["prefill"])
    assert out_dbl["gap"] == out_base["gap"] and out_dbl["fixed"] == out_base["fixed"]


def test_llama_fixed_cost_is_the_ledger_prior():
    p = {**PROFILE, "fixed_call_s": {"hq": 0.1}}
    assert W.fixed_for(p, "llama") == W.LLAMA_FIXED_CALL_S and W.fixed_for(p, "hq") == 0.1


def test_two_agents_take_the_slower_stream():
    a = [[1000, 1000, 100, 0, 0.0, 1.0]] * 2
    b = [[1000, 500, 400, 0, 0.0, 1.0]] * 2
    prof = {"fixed_call_s": {"hq": 0.0}, "runs": [{"agents": {"0": a, "1": b}}]}
    w0, parts = W.wall_est_two(prof, 1000.0, 100.0, "hq", rho=0.0)
    assert w0 == pytest.approx(max(2 * (1.0 + 1.0), 2 * (0.5 + 4.0)))
    w1, _ = W.wall_est_two(prof, 1000.0, 100.0, "hq", rho=1.0)
    assert w1 == pytest.approx(max(2 * 2.0 + 1.0, 2 * 4.5 + 2.0))
    assert parts["blocked_prefill"] == 0.0


def synthetic_campaigns(store, n=4, exact=True):
    """HQ-like campaigns whose recorded call durations obey the cost model exactly."""
    cfg = add_cfg(store, ["python", "vllm", "serve", "/m/Swift-1.5-Qwen3.8-27B-W4A16-codexh", "--max-num-seqs", "1", "--max-model-len", "102400",
                          "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", "0.86",
                          "--speculative-config", '{"num_speculative_tokens": 5}'])
    pf, dc, fx = 1000.0, 100.0, 0.1
    tasks = ["rust/doubly-linked-list", "rust/book-store", "rust/react", "rust/alphametics", "rust/forth", "rust/xorcism",
             "rust/nucleotide-codons", "javascript/complex-numbers"]
    for i in range(n):
        tdict, wall = {}, 0.0
        for t in tasks:
            calls = []
            for j in range(3 + i):
                new, out = 2000 + 100 * j + 50 * i, 300 + 40 * j + 10 * i
                dur = fx + new / pf + out / dc
                calls.append([10000 + 3000 * j, new, out, 1 if j else 0, 1.5 if j else 0.0, dur])
            tdict[t] = calls
            tw = sum(c[5] + c[4] for c in calls) + 5.0
            wall += tw
            add_m(store, cfg, "task_wall_s", tw, item=f"c{i}", task=t)
        add_m(store, cfg, "prefill_tok_s", pf, item=f"c{i}")
        add_m(store, cfg, "decode_tok_s", dc, item=f"c{i}")
        add_m(store, cfg, "wall_s", wall, item=f"c{i}")
        store.upsert("workloads", {"id": f"calls:c{i}", "kind": "campaign-calls", "campaign": f"c{i}", "config_id": cfg["config_id"],
                                   "summary_sha256": f"c{i}", "workload_id": "codex-8task-1agent", "valid": True, "tasks": tdict,
                                   "wall_s": wall})
    return cfg


def test_profile_build_recovers_constants_and_calibration_closes(tmp_path):
    s = Store(tmp_path)
    synthetic_campaigns(s, n=4)
    profiles = W.build_profiles(s, write=True)
    prof = next(p for p in profiles if p["id"] == "profile:codex-8task-1agent")
    assert prof["n_campaigns"] == 4 and prof["fixed_call_s"]["hq"] == pytest.approx(0.1, abs=1e-9)
    assert prof["task_overhead_s"] == pytest.approx(5.0, abs=1e-9)
    assert s.tables["workloads"]["profile:codex-8task-1agent"]["n_campaigns"] == 4
    res = W.calibrate(s, profiles)
    a = next(v for k, v in res["tiers"].items() if k.startswith("A "))
    assert a["n"] == 4 and a["max_abs_pct"] < 1e-6


def test_invalid_campaign_is_not_pooled(tmp_path):
    s = Store(tmp_path)
    synthetic_campaigns(s, n=3)
    w = s.tables["workloads"]["calls:c1"]
    s.upsert("workloads", {**w, "valid": False})
    prof = next(p for p in W.build_profiles(s, write=False) if p["id"] == "profile:codex-8task-1agent")
    assert prof["n_campaigns"] == 2 and "c1" not in prof["sources"]


def test_leave_one_out_tier_needs_other_configs(tmp_path):
    s = Store(tmp_path)
    synthetic_campaigns(s, n=3)
    res = W.calibrate(s)
    b = next(v for k, v in res["tiers"].items() if k.startswith("B ("))
    assert b is None  # all campaigns share one config: nothing independent to pool


def test_two_agent_profile_reproduces_recorded_wall_on_real_runs(fx, db):
    run([fx / "two-agent-hq-v3", fx / "two-agent-llama-v3", fx / "campaign-hq-lost4"], db=db)  # the HQ campaign fits the fixed cost
    s = Store(db)
    profiles = W.build_profiles(s, write=False)
    res = W.calibrate(s, profiles)
    rows = res["two_agent"]
    assert len(rows) == 2
    assert all(abs(r["err_rho0_pct"]) < 3 for r in rows)
    assert all(r["err_rho1_pct"] > r["err_rho0_pct"] for r in rows)
    # rho was fitted on these runs, so the tie width is the rho 0..1 span, not the ~1% in-sample error
    assert res["held_out_pct"]["two-agent"] >= max(abs(r["err_rho1_pct"]) for r in rows) > 5


def test_real_hq_call_tables_tier_a_error_is_small(fx, db):
    run([fx / "campaign-hq-lost4", fx / "campaign-hq-1task"], db=db)
    s = Store(db)
    res = W.calibrate(s)
    a = next(v for k, v in res["tiers"].items() if k.startswith("A "))
    assert a["n"] == 2 and a["max_abs_pct"] < 10
