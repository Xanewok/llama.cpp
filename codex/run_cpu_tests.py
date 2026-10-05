import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1",
                  TMPDIR=str(root / "codex/tmp"), LLAMA_CACHE=str(root / "codex/tmp"))
sys.path.insert(0, str(root / "tools/server/tests"))
import utils
import pytest

os.environ["LD_LIBRARY_PATH"] = str((root / sys.argv[1]).parent)

# All model resolution and downloads are bypassed for this offline run.
utils.ServerPreset.load_all = staticmethod(lambda: None)
preset = utils.ServerPreset.tinyllama2

def local_preset():
    server = preset()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = "/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad/llama-cache/models--ggml-org--test-model-stories260K/snapshots/479896ec924af6d40fd419ab8f4d1eb2101de00d/stories260K-f32.gguf"
    server.n_threads = 8
    server.server_path = str(root / sys.argv[1])
    server.server_port = 18179
    server.log_path = str(root / "codex/server-test.log")
    return server

utils.ServerPreset.tinyllama2 = staticmethod(local_preset)
raise SystemExit(pytest.main([str(root / "tools/server/tests/unit/test_completion.py"),
                             "--basetemp", str(root / "codex/tmp/pytest"), *sys.argv[2:]]))
