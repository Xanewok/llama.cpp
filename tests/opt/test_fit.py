import pytest

from scripts.opt import fit
from scripts.opt.store import Store

from .synth import add_cfg, add_m, llama_cmd, vram_points


def build(tmp_path):
    s = Store(tmp_path)
    for np_, ctx in ((1, 102400), (2, 204800)):
        for k in (3, 5):
            vram_points(s, "IQ4_XS", "q8_0", k, np_, ctx, 8000, 8600, 0.025)
    return s


def test_recovers_a_b_c_from_exact_data(tmp_path):
    m = fit.fit_llama(build(tmp_path))
    g = next(iter(m["groups"].values()))
    assert g["b"] == pytest.approx(0.025, rel=1e-6) and g["b_source"] == "fitted"
    assert g["a"][0] == pytest.approx(8000) and g["a"][1] == pytest.approx(8600)
    assert m["c"] == pytest.approx(100.0)
    assert max(abs(p["resid"]) for p in m["points"]) < 1e-6


def test_noise_shows_up_as_residual_and_stale_flag(tmp_path):
    s = build(tmp_path)
    cfg = vram_points(s, "IQ4_XS", "q8_0", 4, 2, 204800, 8000, 8600, 0.025)
    add_m(s, cfg, "vram_min_free_mib", 3000.0, gpu=0, np=2, ctx=204800, item="outlier")
    m = fit.fit_llama(s)
    assert any(fit.stale(p) for p in m["points"])
    assert max(abs(p["resid"]) for p in m["loo"]) > 100


def test_slope_is_not_fitted_from_nearly_equal_contexts(tmp_path):
    s = Store(tmp_path)
    vram_points(s, "IQ4_XS", "f16", 3, 2, 198144, 9000, 9600, 0.04)
    vram_points(s, "IQ4_XS", "f16", 3, 2, 204800, 9000, 9600, 0.04)
    vram_points(s, "Q5_K_S", "q8_0", 3, 1, 102400, 6000, 6600, 0.025)
    vram_points(s, "Q5_K_S", "q8_0", 3, 2, 204800, 6000, 6600, 0.025)
    m = fit.fit_llama(s)
    f16 = next(g for k, g in m["groups"].items() if k[2] == "f16/f16")
    assert f16["b_source"] == "scaled by KV bytes"
    assert f16["b"] == pytest.approx(0.025 * 2.0 / 1.0625, rel=1e-6)


def test_split_and_ub_are_group_identity(tmp_path):
    s = Store(tmp_path)
    a = add_cfg(s, llama_cmd(split="1,1"))
    b = add_cfg(s, llama_cmd(split="69,31"))
    c = add_cfg(s, llama_cmd(ub=512))
    assert len({fit.group_key(x) for x in (a, b, c)}) == 3


def test_predict_free_and_bands(tmp_path):
    s = build(tmp_path)
    m = fit.fit_llama(s)
    cfg = add_cfg(s, llama_cmd("IQ4_XS", 2, 204800, "q8_0", 4))
    free, basis = fit.predict_free(m, cfg)
    assert free == pytest.approx(8000 - 0.025 * 204800 - 100 * 4)
    assert fit.classify(100) == "reject" and fit.classify(300) == "probe-needed" and fit.classify(900) == "fits"
    assert fit.predict_free(m, add_cfg(s, llama_cmd("J5q", 1, 102400, "f16"))) is None


def test_hq_pool_rule_and_prefix_state():
    pool0 = {"hq:codexh": 199752}
    assert fit.hq_pool(pool0, 4355, "hq:codexh", 8, 0.86) == 199752 - 3 * 4355
    assert fit.hq_pool(pool0, 4355, "hq:codexh", 5, 0.88) == 199752 + 17200
    assert fit.hq_prefix_state(199033, 60000) == "kept"
    assert fit.hq_prefix_state(199033, 61500) == "boundary"
    assert fit.hq_prefix_state(199033, 62747) == "lost" and fit.hq_prefix_state(199033, 70000) == "lost"


def hq_cmd(ckpt, k, util=0.86, np_=1):
    return ["python", "vllm", "serve", f"/m/Swift-1.5-Qwen3.8-27B-W4A16-{ckpt}", "--kv-cache-dtype", "fp8", "--gpu-memory-utilization", str(util),
            "--max-num-seqs", str(np_), "--max-model-len", "102400",
            "--speculative-config", '{"method": "mtp", "num_speculative_tokens": %d}' % k]


def test_hq_pool_slope_is_fitted_from_two_k_values(tmp_path):
    s = Store(tmp_path)
    for ckpt, k, pool in (("attn-int8", 5, 199752), ("attn-int8", 8, 199752 - 3 * 4355), ("codexh", 5, 199752), ("syv-fast", 5, 248788)):
        cfg = add_cfg(s, hq_cmd(ckpt, k))
        assert cfg["engine"] == "hq" and cfg["k"] == k and cfg["gpu_util"] == 0.86
        add_m(s, cfg, "kv_pool_tokens", pool, item=f"{ckpt}{k}")
    hq = fit.fit_hq(s)
    assert hq["per_k"] == pytest.approx(4355) and not hq["per_k_prior"]
    assert max(abs(r["resid"]) for r in hq["rows"]) < 1e-6
    assert fit.hq_pool(hq["pool0"], hq["per_k"], "swift-1.5:W4A16-syv-fast", 5, 0.88) == 248788 + 17200


def test_prefix_rule_against_recorded_probes(fx, db):
    from scripts.opt.ingest import run
    run([fx / "hq-conc"], db=db)
    s = Store(db)
    hq = fit.fit_hq(s)
    assert hq["pools"]
    chk = fit.check_prefix_rule(s, hq)
    assert {c["rule"] for c in chk} >= {"kept", "lost"}
    assert all((c["rule"] == "kept") == (c["obs"] == "kept") for c in chk if c["rule"] != "boundary")


def test_report_text_contains_residual_table(tmp_path):
    text, parts = fit.report(build(tmp_path))
    assert "leave-one-boot-cell-out" in text and "in-sample" in text and parts["llama"]["groups"]
