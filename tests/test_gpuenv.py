from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.instruments import emit
from scripts.instruments import gpuenv

ONE_CARD = "NVIDIA GeForce RTX 3090, 580.173.02, 390.00 W\n"
TWO_CARDS = ONE_CARD + "NVIDIA GeForce RTX 3090, 580.173.02, 390.00 W\n"
NVCC = "Cuda compilation tools, release 13.0, V13.0.88\n"


class GpuEnvParseTest(unittest.TestCase):
    def test_parses_a_single_card(self) -> None:
        env = gpuenv.parse(ONE_CARD, NVCC)
        self.assertEqual(env["gpu"], "NVIDIA GeForce RTX 3090")
        self.assertEqual(env["gpu_count"], 1)
        self.assertEqual(env["gpu_driver"], "580.173.02")
        self.assertEqual(env["power_limit_w"], 390.0)
        self.assertEqual(env["cuda_toolkit"], "13.0.88")

    def test_counts_identical_cards(self) -> None:
        self.assertEqual(gpuenv.parse(TWO_CARDS, NVCC)["gpu_count"], 2)

    def test_refuses_mixed_cards(self) -> None:
        mixed = ONE_CARD + "NVIDIA GeForce RTX 5070 Ti, 580.173.02, 300.00 W\n"
        with self.assertRaises(ValueError):
            gpuenv.parse(mixed, NVCC)

    def test_refuses_one_card_model_at_two_power_caps(self) -> None:
        # The cap is the experiment variable; averaging or picking the first would
        # attribute a number to a wattage the card never ran at.
        skewed = ONE_CARD + "NVIDIA GeForce RTX 3090, 580.173.02, 350.00 W\n"
        with self.assertRaises(ValueError):
            gpuenv.parse(skewed, NVCC)

    def test_mixed_capture_preserves_each_card_and_cap(self) -> None:
        mixed = ONE_CARD + "NVIDIA GeForce RTX 5070 Ti, 580.173.02, 300.00 W\n"
        env = gpuenv.parse(mixed, NVCC, allow_mixed=True)
        self.assertEqual(env["gpu_count"], 2)
        self.assertIsNone(env["power_limit_w"])
        self.assertEqual([d["power_limit_w"] for d in env["gpu_devices"]], [390.0, 300.0])
        self.assertIn("RTX 5070 Ti", env["gpu"])

    def test_cuda_visibility_requires_explicit_nvml_selection(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaises(SystemExit):
                gpuenv.capture()

    def test_device_selection_reaches_nvml(self) -> None:
        with patch.object(gpuenv, "_run", side_effect=[ONE_CARD, NVCC]) as run:
            self.assertEqual(gpuenv.capture("GPU-test")["gpu_count"], 1)
        self.assertIn("--id=GPU-test", run.call_args_list[0].args[0])

    def test_missing_nvcc_leaves_the_toolkit_unknown(self) -> None:
        # nvidia-smi's "CUDA Version" is the driver API level, not the toolkit, so
        # there is no second source to fall back on.
        self.assertIsNone(gpuenv.parse(ONE_CARD, None)["cuda_toolkit"])

    def test_absent_power_reading_is_null_not_zero(self) -> None:
        env = gpuenv.parse("NVIDIA GeForce RTX 3090, 580.173.02, [N/A]\n", NVCC)
        self.assertIsNone(env["power_limit_w"])

    def test_no_gpus_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            gpuenv.parse("\n", NVCC)


class EnvChannelTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def context(self, lane: str, *extra: str) -> emit.RunContext:
        parser = argparse.ArgumentParser()
        emit.add_context_arguments(parser)
        args = parser.parse_args(
            ["--lane", lane, "--quant", "Q4", "--run-index", "0",
             "--out", str(self.dir / "out.jsonl"), *extra]
        )
        return emit.make_context(args)

    def write_env(self, **fields: object) -> Path:
        path = self.dir / "env.json"
        path.write_text(json.dumps({"gpu": "NVIDIA GeForce RTX 3090", **fields}))
        return path

    def test_capture_reaches_the_row(self) -> None:
        env = self.write_env(gpu_count=2, power_limit_w=390.0, gpu_driver="580.173.02")
        row = emit.build_row(self.context("quant", "--env-json", str(env)), "t", 0)
        self.assertEqual(row["gpu"], "NVIDIA GeForce RTX 3090")
        self.assertEqual(row["gpu_count"], 2)
        self.assertEqual(row["power_limit_w"], 390.0)
        self.assertEqual(row["schema_version"], emit.SCHEMA_VERSION)

    def test_mixed_devices_reach_the_row(self) -> None:
        captured = gpuenv.parse(ONE_CARD + "RTX 2060 SUPER, 580.173.02, 175 W\n",
                                NVCC, allow_mixed=True)
        path = self.dir / "mixed.json"
        path.write_text(json.dumps(captured))
        row = emit.build_row(self.context("quant", "--env-json", str(path)), "t", 0)
        self.assertEqual(row["gpu_devices"], captured["gpu_devices"])
        self.assertIsNone(row["power_limit_w"])

    def test_local_lane_without_env_refuses(self) -> None:
        with self.assertRaises(SystemExit):
            self.context("quant")

    def test_unreadable_env_refuses(self) -> None:
        with self.assertRaises(SystemExit):
            self.context("quant", "--env-json", str(self.dir / "absent.json"))

    def test_env_naming_no_gpu_refuses(self) -> None:
        path = self.dir / "env.json"
        path.write_text(json.dumps({"gpu": None, "gpu_count": 1}))
        with self.assertRaises(SystemExit):
            self.context("quant", "--env-json", str(path))

    def test_hand_edited_field_refuses(self) -> None:
        # A field gpuenv.py does not produce means the file was written by hand,
        # which is the labeling channel this replaces.
        path = self.dir / "env.json"
        path.write_text(json.dumps({"gpu": "RTX 3090", "watts": 390}))
        with self.assertRaises(SystemExit):
            self.context("quant", "--env-json", str(path))

    def test_frontier_lane_needs_no_gpu(self) -> None:
        self.assertIsNone(self.context("frontier").gpu)

    def test_no_hosted_lane_demands_a_gpu_capture(self) -> None:
        # An API lane has no card to capture, and the guard used to name only two lanes,
        # so every routed lane exited at launch asking for env.json.
        for lane in emit.LANE_MODELS:
            if lane in emit.LOCAL_LANES:
                continue
            self.assertIsNone(self.context(lane).gpu, lane)



class CodexEndpointTest(unittest.TestCase):
    """The probe must interrogate the endpoint Codex actually calls.

    A leg was served on 8010 while Codex was pinned to 8000. The quant guard confirmed a
    healthy server on 8010 and all 40 tasks then died calling 8000, so the run came back
    0/40 with rc=0 and every row blaming the model.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)

    def write(self, provider: str = "vast", url: str = "http://127.0.0.1:8000/v1") -> None:
        (self.home / "qwen38.config.toml").write_text(f'model_provider = "{provider}"\n')
        (self.home / "config.toml").write_text(
            f'[model_providers.{provider}]\nbase_url = "{url}"\n'
        )

    def test_resolves_through_profile_and_provider(self) -> None:
        self.write()
        self.assertEqual(
            emit.codex_base_url(home=self.home), "http://127.0.0.1:8000/v1"
        )

    def test_missing_config_is_none_not_a_guess(self) -> None:
        self.assertIsNone(emit.codex_base_url(home=self.home))

    def test_unknown_provider_is_none(self) -> None:
        (self.home / "qwen38.config.toml").write_text('model_provider = "absent"\n')
        (self.home / "config.toml").write_text('[model_providers.vast]\nbase_url = "x"\n')
        self.assertIsNone(emit.codex_base_url(home=self.home))

    def test_the_real_config_still_resolves(self) -> None:
        # Guards against a Codex config change silently disabling the check.
        url = emit.codex_base_url()
        self.assertTrue(url is None or url.startswith("http"), url)

class ParallelLegProfileTest(unittest.TestCase):
    """Two legs at once need two local ports, and a profile is what picks one.

    A profile pins a provider and a provider pins one base_url, so every leg used to queue
    behind 8000. The guard that catches a served/called mismatch reads the same profile, so
    it must follow the selection rather than stay on the default.

    Driven through a subprocess rather than importlib.reload: reload mutates the module in
    place, so restoring the environment afterwards also rewrites the object under test and
    the assertions read the default no matter what was selected.
    """

    def probe(self, profile: str | None, expr: str) -> str:
        env = {**os.environ}
        env.pop("QWEN38_CODEX_PROFILE", None)
        if profile is not None:
            env["QWEN38_CODEX_PROFILE"] = profile
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.')\nfrom scripts.instruments import emit; print(" + expr + ")"],
            capture_output=True, text=True, timeout=60, env=env,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout.strip()

    def test_the_default_is_unchanged(self) -> None:
        self.assertEqual(self.probe(None, "emit.CODEX_PROFILE"), "qwen38")

    def test_the_selected_profile_reaches_the_codex_command(self) -> None:
        self.assertIn("qwen38b", self.probe("qwen38b", "emit.codex_command('quant','xhigh')"))

    def test_each_profile_resolves_to_its_own_port(self) -> None:
        # Two profiles on one port would silently share a tunnel, and each leg would
        # measure whichever server won the race.
        urls = {p: self.probe(p, "emit.codex_base_url()")
                for p in ("qwen38", "qwen38b", "qwen38c")}
        present = {p: u for p, u in urls.items() if u and u != "None"}
        self.assertEqual(len(present), 3, urls)
        self.assertEqual(len(set(present.values())), 3, present)


class ToolOutputCapTest(unittest.TestCase):
    """The cap is a leg variable, so it must be absent unless a leg asks for it."""

    def probe(self, value: str | None) -> str:
        env = {**os.environ}
        env.pop("QWEN38_TOOL_OUTPUT_TOKEN_LIMIT", None)
        if value is not None:
            env["QWEN38_TOOL_OUTPUT_TOKEN_LIMIT"] = value
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.')\nfrom scripts.instruments import emit; "
             "print(' '.join(emit.codex_command('glm-flash','xhigh')))"],
            capture_output=True, text=True, timeout=60, env=env,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    def test_absent_by_default(self) -> None:
        self.assertNotIn("tool_output_token_limit", self.probe(None))

    def test_present_when_the_leg_sets_it(self) -> None:
        self.assertIn("tool_output_token_limit=3000", self.probe("3000"))


class ApiLaneTest(unittest.TestCase):
    """The lanes the local card is measured against.

    Naming the provider is not optional: a slash-bearing id with no provider goes to
    Codex's default, which answers happily and produces rows labelled deepseek that
    were served by something else.
    """

    def test_a_routed_lane_names_its_provider(self) -> None:
        command = emit.codex_command("deepseek-flash", "xhigh")
        self.assertIn("deepseek/deepseek-v4-flash", command)
        self.assertIn("model_provider=openrouter", command)

    def test_a_first_party_lane_does_not(self) -> None:
        self.assertNotIn("model_provider=openrouter", emit.codex_command("frontier", "xhigh"))

    def test_the_local_lane_still_uses_its_profile(self) -> None:
        command = emit.codex_command("quant", "xhigh")
        self.assertIn("-p", command)
        self.assertNotIn("model_provider=openrouter", command)

    def test_both_tiers_are_reachable_and_distinct(self) -> None:
        self.assertNotEqual(emit.lane_model("deepseek-flash"), emit.lane_model("deepseek-pro"))

    def test_every_routed_lane_names_its_provider(self) -> None:
        # A lane added without the provider flag reaches Codex's default silently, so the
        # check is over the whole table rather than the two lanes that existed when it
        # was written.
        for lane, model in emit.LANE_MODELS.items():
            if "/" in model:
                command = emit.codex_command(lane, "xhigh")
                self.assertIn(model, command, lane)
                self.assertIn("model_provider=openrouter", command, lane)

    def test_the_new_cheap_tiers_are_distinct_routes(self) -> None:
        ids = {lane: emit.lane_model(lane)
               for lane in ("deepseek-flash", "deepseek-flash-vision", "glm-flash")}
        self.assertEqual(len(set(ids.values())), 3, ids)

    def test_every_lane_in_the_table_is_launchable(self) -> None:
        # The gap this closes: emit knew the deepseek lanes, argparse did not, so they were
        # unit-tested and still rejected at the command line.
        parser = argparse.ArgumentParser()
        emit.add_context_arguments(parser)
        for lane in emit.LANE_MODELS:
            args = parser.parse_args(["--lane", lane, "--quant", "x", "--run-index", "0",
                                      "--out", "/tmp/x.jsonl"])
            self.assertEqual(args.lane, lane)

    def test_an_unknown_lane_still_refuses(self) -> None:
        with self.assertRaises(ValueError):
            emit.lane_model("deepseek-v9")


if __name__ == "__main__":
    unittest.main()
