"""CPU stub of the whole KLD item (GPUs hidden): preflight, canary, run, analysis through run.sh with the stub dump, then the verdict's
negative controls. Writes only under OUT.  usage: stub_e2e.py OUT
"""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import verdict  # noqa: E402


def token_file(path, seqs):
    with open(path, "wb") as f:
        f.write(b"KLDT" + struct.pack("<i", len(seqs)))
        for t in seqs:
            f.write(struct.pack("<i", len(t)) + np.asarray(t, "<i4").tobytes())


def main(out):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    fx = out / "fixture"
    fx.mkdir(parents=True, exist_ok=True)
    t = np.array([248045, 74455, 198] + list(range(33)) + [248046], dtype="<i4")
    token_file(fx / "a.bin", [t, t])
    token_file(fx / "b.bin", [t, t, t])
    token_file(fx / "canary.bin", [np.array([248045, 74455, 198] + list(range(100)) + [248046], dtype="<i4")])
    arms = ["base", "cand", "cand_repeat", "repeat"]
    stub_build = dict(root="/x", bin="/x", commit="s", binary="/x/kld_dump", binary_sha256="0")
    config = dict(root=str(ROOT), scratch=str(fx / "scratch"), model="m", model_sha="0", tokens=str(fx / "a.bin"),
                  canary=str(fx / "canary.bin"), canary_source=0, canary_max_nll=30, builds=dict(base=stub_build, cand=stub_build),
                  sha256={}, verdict=dict(all_tail=dict(expected=3e-5, falsifier=3e-4, ratio=10.0)),
                  depths=[dict(name="da", tokens=str(fx / "a.bin"), selection=[0, 1], prefix=None, tail=12, arms=arms),
                          dict(name="db", tokens=str(fx / "b.bin"), selection=[0, 1, 0], prefix=None, tail=12, arms=arms + ["f16"])])
    (fx / "config.json").write_text(json.dumps(config))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    for stage, sub in (("preflight", "preflight"), ("canary", "canary"), ("run", "run"), ("analysis", "run")):
        cmd = ["bash", str(ROOT / ("preflight.sh" if stage == "preflight" else "run.sh"))] + ([] if stage == "preflight" else [stage])
        cmd += ["--stub", "--config", str(fx / "config.json"), "--out", str(out / sub), "--budget-s", "300"]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, (stage, p.stdout, p.stderr)
    # a GPU stage refuses a config outside gen/
    p = subprocess.run(["bash", str(ROOT / "run.sh"), "run", "--config", str(fx / "config.json"), "--out", str(out / "refused")],
                       env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 6, (p.returncode, p.stderr)
    r = json.loads((out / "run/SUMMARY.json").read_text())
    db = r["depths"]["db"]
    v = db["verdict_all_tail"]
    assert v["f16_control_mean_kl"] > v["mean_kl_candidate_to_base"] > 0 and not v["falsified"], v
    assert r["depths"]["da"]["verdict_all_tail"]["f16_control_mean_kl"] is None
    kl = dict(mean=v["f16_control_mean_kl"] * 2)
    over = verdict(kl, [0.0, 0.0], expected=1.0, falsifier=1.0, f16=v["f16_control_mean_kl"])
    assert over["exceeds_f16_control"] and not over["falsified"], over
    assert verdict(dict(mean=4e-4), [0.0, 0.0], expected=3e-5, falsifier=3e-4)["falsified"]
    assert not verdict(dict(mean=1e-5), [0.0, 0.0], expected=3e-5, falsifier=3e-4)["falsified"]
    print("kld stub: preflight canary run analysis PASS; verdict controls (absolute falsifier, f16 recorded only, gen/ config) PASS")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
