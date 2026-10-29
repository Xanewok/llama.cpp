"""Swift IQ4_XS eight-task long tail on the 5060 Ti x2 pair, reusing swift_longtail unchanged.

Deltas from campaign-swift-iq4-xs-longtail-20260914-rerun2 (5060 Ti + 2060 SUPER), and only these:
second device UUID, tensor split 50,50 (equal cards), both power limits 150 W (the 5060 Ti floor;
the 2060 SUPER ran at 140 W), and 1Password's GPU process tolerated as desktop load.
"""

import os
import shlex
import sys
from pathlib import Path

from scripts.instruments import hillclimb as h
from scripts.instruments import swift_longtail as s

GPU1 = "GPU-3fe72007-d2f3-11c0-4d57-cc6fbde1451f"
SPLIT = "50,50"
MAXIMUM = int(os.environ.get("SWIFT_DRAFT_MAX", "3"))  # --spec-draft-n-max; 3 matches the reference runs
CAPS = (150, 150)

if os.environ.get("SWIFT_ENGINE") == "upstream":
    # Built with the llama.cpp CMakeUserPresets settings: CUDA 13.2 libs, NCCL off (internal AllReduce).
    h.LLAMA = Path("/home/xanewok/repos/llama.cpp-upstream")
    h.COMMIT = "ed7ac35e1ee49cb70e4dfa9f0a2ce39b0a5ec4ea"
elif os.environ.get("SWIFT_ENGINE") == "mtp-fast":
    # Branch swift-mtp-fast: upstream ed7ac35e1 + PR #27694, backend sampling under -sm tensor, ik#1736 fused catch-up.
    # Same build settings as upstream. Draft GPU sampling is on by default; the rest are flags (SWIFT_EXTRA).
    h.LLAMA = Path("/home/xanewok/repos/llama.cpp-mtp")
    h.COMMIT = "3ec79cbfa0ab6ba291ed98c1f2afb7c1fcdd6504"
elif os.environ.get("SWIFT_ENGINE") == "frozen":
    # A frozen copy of a branch build (SP/combo/freeze.sh): build/bin with RUNPATH $ORIGIN, gguf-py, and a git dir
    # detached at the source commit; the manifest's binary_pins guard the bytes.
    h.LLAMA = Path(os.environ["SWIFT_LLAMA"])
    h.COMMIT = os.environ["SWIFT_COMMIT"]
h.SERVER_EXTRA[:] = shlex.split(os.environ.get("SWIFT_EXTRA", ""))  # recorded as manifest server_extra
SWIFT15 = {  # repo ukisai/Swift-1.5-Qwen3.8-27B-GGUF; sha256 of the local file
    "swift15-iq4-xs": ("Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf", 15475951296,
                       "e3b59faa312a6202d7ac0a2400123956104bc720e77f6ff21c754aec5c3740df"),
    "swift15-q5-k-m": ("Swift-1.5-Qwen3.8-27B-Q5_K_M.gguf", 20923877056,
                       "4964843f816dafe77ce24a92c97f3bb094b218d752172710d9438af0e2562931"),
    "swift15-q6-k": ("Swift-1.5-Qwen3.8-27B-Q6_K.gguf", 23860565696,
                     "cd2eb6597026ce4eb86100e830be24f01461f84b89d4708634c2ec3d43b67fa8"),
    # J1: own requant, not in the ukisai repo (bartowski IQ4_XS tensor map + imatrix, ssm_out -> Q6_K; SP/vast-gguf-j/RECIPE.txt)
    "swift15-iq4-xs-j1": ("Swift-1.5-Qwen3.8-27B-IQ4_XS-ssmout6.gguf", 15792490752,
                          "90a3b0e5d6c720c198ab38a5b05f14458870326c58800ae08b2e79df892fc1c2"),
    # quant-ladder download (SP/quant-ladder/dl.log, sha256 checked there); lives on /mnt/scratch, /home has no room
    "swift15-q5-k-s": ("Swift-1.5-Qwen3.8-27B-Q5_K_S.gguf", 19566913216,
                       "9dc2d575b1d241ba5dd74321a330c9fcd2bb7c1c5310f1c05fcdaa1428e087fc"),
}
LOCAL_BUILDS = {"swift15-iq4-xs-j1": {"repo": "local:gguf-j J1 (from ukisai Swift 1.5 BF16)", "revision": "J1"}}
MODEL_DIRS = {"swift15-q5-k-s": Path("/mnt/scratch/work/models/gguf")}
if os.environ.get("SWIFT_MODEL"):
    model_id = os.environ["SWIFT_MODEL"]
    file, size, digest = SWIFT15[model_id]
    s.MODEL_PATH = MODEL_DIRS.get(model_id, Path.home() / "models") / file
    s.MODEL = {"id": model_id, "repo": "ukisai/Swift-1.5-Qwen3.8-27B-GGUF",
               "revision": "a1614465cfa35d04d3e8575d713fa779662b5eab", "file": file, "bytes": size,
               "sha256": digest, "local": str(s.MODEL_PATH)}
    s.MODEL.update(LOCAL_BUILDS.get(model_id, {}))
    h.TEMPLATE_OVERRIDE_MODELS.add(model_id)  # same embedded template as Swift 1.0 (sha256 c3cf9e34...)
h.GPU1 = GPU1
h.DEVICES[1] = GPU1  # mutate in place: agentic_campaign holds a reference to this list
h.DISPLAY_COMPUTE_NAMES.add("1password")


def safety_check(sample, reserves=True, vram_reserves=(512, 256)):
    if sample["ram_available_mib"] < 2048:
        raise h.StopCampaign("system_ram_below_2gib")
    if len(vram_reserves) != 2 or any(type(value) is not int or value < 0 for value in vram_reserves):
        raise h.StopCampaign("invalid_vram_reserves")
    for device, cap, reserve in zip(h.DEVICES, CAPS, vram_reserves):
        gpu = sample["gpus"][device]
        if gpu["temperature.gpu"] >= 80:
            raise h.StopCampaign("thermal_limit: " + device)
        if abs(gpu["power.limit"] - cap) > 0.1:
            raise h.StopCampaign("power_limit_changed: " + device)
        if reserves and gpu["memory.free"] < reserve:
            raise h.RejectArm("vram_reserve: " + device)


h.safety_check = safety_check

if os.environ.get("SWIFT_TASKS"):  # comma-separated subset for repeats; in place, s reads the list at call time
    s.LONG_TAIL_TASKS[:] = os.environ["SWIFT_TASKS"].split(",")

_queue = s.queue
s.queue = lambda: [{**job, "configs": [{**job["configs"][0], "split": SPLIT, "maximum": MAXIMUM}]} for job in _queue()]

if __name__ == "__main__":
    sys.exit(s.main())
