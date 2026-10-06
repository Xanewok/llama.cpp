"""Break each production change, require a failing control, then restore and rebuild."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "codex/host-results"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
GATE = ROOT / "codex/host-gate.sh"
BUILDGATE = "/home/xanewok/repos/qwen38-eval-worktrees/bin/buildgate"
env = dict(os.environ, CCACHE_READONLY="1", CCACHE_NOSTATS="1", CCACHE_TEMPDIR=str(ROOT / "build-tmp/ccache"),
           LD_LIBRARY_PATH=str(ROOT / "build-cpu/bin"), LLAMA_HOST_TRIMS_PROOF="1", GGML_SCHED_INPUT_STAGING="1")
results = []


def command(tag, args, check=True):
    with (OUT / (tag + ".log")).open("w") as log:
        return subprocess.run(["bash", str(GATE), *args], cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=check).returncode


def rebuild(tag):
    command(tag, [BUILDGATE, "cmake", "--build", "build-cpu", "--target", "llama-server", "test-kv-mask", "-j4"])


for name, relative, old, new in (
    ("kv-bound", "src/llama-kv-cache.cpp", "it->first < p1", "it->first <= p1"),
    ("kv-head", "src/llama-kv-cache.cpp", "new_head = std::min(new_head, i);", "if (new_head == cells.size()) { new_head = i; }"),
    ("mtp-offset", "common/speculative.cpp", "h_tgt + (size_t) i_batch_beg[seq_id] * n_embd, (size_t) n_rows * row_bytes",
     "h_tgt, (size_t) n_rows * row_bytes"),
):
    path = ROOT / relative
    original = path.read_text()
    assert original.count(old) == 1, name
    try:
        path.write_text(original.replace(old, new))
        rebuild("build-mutant-" + name)
        if name.startswith("kv-"):
            rc = command("negative-" + name, [str(ROOT / "build-cpu/bin/test-kv-mask")], check=False)
            assert rc != 0, f"{name}: production mutant escaped KV oracle"
            reason = (OUT / ("negative-" + name + ".log")).read_text().splitlines()[-1]
        else:
            mutant = OUT / "negative-mtp-offset.json"
            command("negative-mtp-offset", [str(ROOT / "build-cpu/bin/host-identity"),
                "/mnt/scratch/work/models/reference/Qwen3.5-4B-UD-Q4_K_XL.gguf", str(mutant), "1", "2", "1729", "48"])
            good = json.loads((OUT / "identity/patch-m1-np2-s1729.json").read_text())
            bad = json.loads(mutant.read_text())
            for data in (good, bad):
                data["stats"].sort(key=lambda item: item["slot"])
            pattern = r"draft candidate\s+\d+, pos\s+\d+:\s+(\d+)\s+\(([\d.]+)\)"
            good_draft = re.findall(pattern, (OUT / "identity/patch-m1-np2-s1729.log").read_text())
            bad_draft = re.findall(pattern, (OUT / "negative-mtp-offset.log").read_text())
            assert good != bad or good_draft != bad_draft, "MTP row-offset mutant escaped identity/draft oracle"
            reason = dict(target_or_counters_differ=good != bad, draft_candidates_differ=good_draft != bad_draft)
        results.append(dict(mutant=name, status="rejected", reason=reason))
        print(f"PASS negative control {name}: {reason}", flush=True)
    finally:
        path.write_text(original)
        rebuild("build-restored-" + name)
(OUT / "negative-controls.json").write_text(json.dumps({"status": "PASS", "controls_M": len(results), "results": results}, indent=2) + "\n")
command("test-kv-mask-restored", [str(ROOT / "build-cpu/bin/test-kv-mask")])
