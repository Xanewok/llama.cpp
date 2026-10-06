"""np2-busy and np1 items for the gdn-getter-gate freeze (build window); harness.mkitem owns the queue manifests.

The experiments are the existing gdn-getter ones with the patch arm swapped for the frozen gate build; the up arm stays the
frozen stack6. Nothing is submitted.
"""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
FROZEN = Path.home() / "repos/qwen38-eval-worktrees/frozen/gdn-getter-gate"
EVAL = Path.home() / "repos/qwen38-eval"
SRC = EVAL / "harness/experiments"
DAY = datetime.date.today().isoformat()

CPU_EVIDENCE = {
    "path": str(ROOT / "codex/bw-results/summary.json"),
    "command": ("bw-getter-gate.sh: cpugate test-recurrent-state-rollback on Qwen3.5-4B-UD-Q4_K_XL.gguf -c 4096: two sequences through prefill, "
                "verify, decode and an idle slot match single-sequence contexts; probe: getter nodes only at n_seqs == 1, gathered state "
                "at n_seqs > 1; row0, nogate and invert mutants fail"),
}

BASE_CANARY = [
    r"(?s)n_kv_pad += 4096\b.*n_kv_pad += 4096\b",
    r"copy-engine H2D streams per GPU: 2\b",
    r"ssm_conv_state_fusion: graph device=0 graph=\d+ tokens=6 conv=([1-9]\d*) fused=\1\b",
    r"ssm_conv_state_fusion: graph device=1 graph=\d+ tokens=6 conv=([1-9]\d*) fused=\1\b",
]
NO_PARTIAL_CONV = r"(?s)\A(?!.*ssm_conv_state_fusion: graph device=\d+ graph=\d+ tokens=[1-8] conv=(\d+) fused=(?!\1\b))"
GETTER_LOG = [rf"gdn_state_indexed: device={d} graph=\d+ tokens=6 gathers_skipped=[1-9]\d*\b" for d in (0, 1)]


def fusion_log(seqs, indexed):
    # fused == gdn: every GDN node of the graph fused its snapshot copy
    return [rf"gdn_cache_fusion: graph device={d} graph=\d+ tokens=6 seqs={seqs} indexed={indexed} gdn=([1-9]\d*) fused=\1\b" for d in (0, 1)]


PATCH_ARM = """[arms.patch]
label = {label}
commit = {commit}
root = {root}
bin = "build/bin"
repo = {repo}
exe = "build/bin/round-trace"
{env}
{flags}
canary_log = [
{canary}
]
"""

VARIANTS = {
    "np2-busy": dict(
        src="gdn-getter-np2-busy-j5q.toml",
        title=("stack6 vs stack6 + GDN state getter gated to single-sequence ubatches (gdn-getter-gate), both --kv-pad 4096, Swift 1.5 J5q TP2 k=5 "
               "-ub 128, two busy streams (-np 2, 2 x 102400) at 8k/32k, ABBA: per-stream and aggregate ms/round; at seqs=2 the getter is off "
               "and the cache-copy fusion stays (gdn_cache_fusion indexed=0 fused=gdn); np2 acceptance and tok/s measure the fixture "
               "(random-token prompts, reduced draft vocab): judge by ms/round between boots with equal per-stream counters"),
        comment=("# Gate under test: getter only at n_seqs == 1, else stack6 gather + fusion. Expected at both slots busy: ms/round equal to stack6 "
                 "(the ungated getter cost +2.2..2.3 ms/round here by losing the fusion).\n"
                 "# np2 acceptance, tok/s and draft counters measure the fixture, not the code: compare ms/round only between boots with equal per-stream counters."),
        label="stack6 + GDN state getter gated to n_seqs == 1 (gdn-getter-gate), --kv-pad 4096, staging",
        canary=BASE_CANARY + fusion_log(2, 0) + [r"(?s)\A(?!.*seqs=2 indexed=[1-9])"],
    ),
    "np1": dict(
        src="gdn-getter-np1-cccl-j5q.toml",
        title=("stack6 vs stack6 + GDN state getter gated to single-sequence ubatches (gdn-getter-gate), both CCCL 3.4.3 (CUB DeviceTopK), "
               "--kv-pad 4096, Swift 1.5 J5q TP2 k=5 np1, ABBA, cross-arm identity: the gate must keep the np1 gain "
               "(-0.48/-0.11/-0.24 ms/round at 8k/32k/90k, token identity)"),
        comment="# At np1 every ubatch has n_seqs == 1, so the gate changes nothing; this is the no-regression and identity check of the gated build.",
        label="stack6 + GDN state getter gated to n_seqs == 1 (gdn-getter-gate), CCCL 3.4.3",
        canary=BASE_CANARY + [NO_PARTIAL_CONV] + GETTER_LOG + [
            rf"gdn_cache_fusion: graph device={d} graph=\d+ tokens=6 seqs=1 indexed=([1-9]\d*) gdn=\1 fused=\1\b" for d in (0, 1)],
    ),
}


def toml_str(value):
    return json.dumps(value)


def line(text, key):
    match = re.search(rf"^{key} = .*$", text, re.M)
    assert match, key
    return match.group(0)


def render_variant(tag, v, commit):
    # the old items' comments describe their own arms
    text = re.sub(r"^#.*\n", "", (SRC / v["src"]).read_text(), flags=re.M)
    name = f"gdn-getter-gate-{tag}-j5q"
    patch_section = re.search(r"^\[arms\.patch\]\n.*?(?=^\[server\])", text, re.M | re.S)
    assert patch_section, "[arms.patch]"
    section = patch_section.group(0)
    arm = PATCH_ARM.format(
        label=toml_str(v["label"]), commit=toml_str(commit), root=toml_str(str(FROZEN)), repo=toml_str(str(ROOT)),
        env=line(section, "env"), flags=line(section, "flags"),
        canary="\n".join(f"  {toml_str(p)}," for p in v["canary"]))
    text = text.replace(section, arm + "\n")
    text = re.sub(r"^id = .*$", f"id = {toml_str(name)}", text, count=1, flags=re.M)
    text = re.sub(r"^title = .*$", lambda _: f"title = {toml_str(v['title'])}\n{v['comment']}", text, count=1, flags=re.M)
    text = re.sub(r"^out = .*$", f"out = {toml_str(str(EVAL / 'results' / f'{DAY}-{name}'))}", text, count=1, flags=re.M)
    cpu = f"cpu_evidence = {{ path = {toml_str(CPU_EVIDENCE['path'])}, command = {toml_str(CPU_EVIDENCE['command'])} }}"
    text = re.sub(r"^cpu_evidence = .*$", lambda _: cpu, text, count=1, flags=re.M)
    pins = [str(ROOT / "codex/bw-getter-gate.sh"), str(ROOT / "tests/test-recurrent-state-rollback.cpp")]
    if "extra_pins = [" in text:
        text = text.replace("extra_pins = [\n", "extra_pins = [\n" + "".join(f"  {toml_str(p)},\n" for p in pins), 1)
    else:
        text = text.replace(cpu, cpu + f"\nextra_pins = [{', '.join(map(toml_str, pins))}]", 1)
    return name, text


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    commit = (FROZEN / "SOURCE").read_text().splitlines()[0]
    items = ROOT / "codex/items"
    items.mkdir(exist_ok=True)
    for tag, v in VARIANTS.items():
        name, text = render_variant(tag, v, commit)
        experiment = items / f"{name}.toml"
        experiment.write_text(text)
        subprocess.run(["python3", "-m", "harness.mkitem", str(experiment), "--out", str(experiment.with_suffix(".gpuq.toml"))],
                       cwd=EVAL, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=True)
        print(f"item {experiment.with_suffix('.gpuq.toml')}")


if __name__ == "__main__":
    main()
