import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.instruments import hillclimb as h
from scripts.instruments import hillclimb_probe as p
from scripts.instruments import swift_longtail as s


class TuningTests(unittest.TestCase):
    def validate(self, jobs):
        with patch.object(h, "MODELS", [s.MODEL]), patch.object(h, "TASKS", s.TUNING_TASKS):
            return h.validate_profiles(jobs)

    def test_baseline_is_exact_matrix_with_output_room(self):
        jobs = self.validate(s.tuning_queue("baseline"))
        self.assertEqual(len(jobs), 18)
        self.assertEqual({(j["task"], j["targets"][0], j["seed"]) for j in jobs},
                         {(t, n, seed) for t in s.TUNING_TASKS for n in (8192, 49152, 98304)
                          for seed in s.TUNING_SEEDS})
        for job in jobs:
            self.assertEqual(job["configs"], [s.TUNING_BASELINE])
            self.assertLessEqual(job["targets"][0] + 2048, job["configs"][0]["context"])

    def test_stages_bracket_baseline_and_retain_selected_settings(self):
        selected = {**s.TUNING_BASELINE, "split": "67,33", "ubatch": 64}
        for stage in s.TUNING_STAGES[1:]:
            jobs = self.validate(s.tuning_queue(stage, selected, splits=["67,33", "71,29"],
                                               microbatches=[64, 256]))
            self.assertIn("baseline-before", jobs[0]["id"])
            self.assertIn("baseline-after", jobs[-1]["id"])
            self.assertEqual(jobs[0]["configs"], [s.TUNING_BASELINE])
            self.assertEqual(jobs[-1]["configs"], [s.TUNING_BASELINE])
        jobs = s.tuning_queue("probability", selected)
        arms = [j for j in jobs if "p_min" in j["id"]]
        self.assertEqual({j["configs"][0]["p_min"] for j in arms}, {0.5, 0.75})
        self.assertTrue(all(j["configs"][0]["split"] == "67,33" for j in arms))

    def test_swarm_context_is_per_slot_and_does_not_truncate_total(self):
        jobs = self.validate(s.tuning_queue("swarm-allocation", s.TUNING_BASELINE))
        arms = [j for j in jobs if j["id"].startswith("swarm-") and "baseline" not in j["id"]]
        self.assertEqual(len(arms), 9)
        self.assertEqual({(j["configs"][0]["slots"], j["configs"][0]["context"]) for j in arms},
                         {(n, c) for n in (2, 3, 4) for c in (32768, 65536, 102400)})

    def test_invalid_controls_and_implicit_winners_fail(self):
        with self.assertRaises(ValueError):
            s.tuning_queue("microbatch")
        for change in ({"p_min": float("nan")}, {"minimum": 4}, {"draft_k": "q4_0"},
                       {"batch": 64}, {"maximum": 0}):
            jobs = copy.deepcopy(s.tuning_queue("baseline"))
            jobs[0]["configs"][0].update(change)
            with self.assertRaises(ValueError):
                self.validate(jobs)

    def test_plan_cli_never_initializes_or_runs_campaign(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(h, "initialize") as init, \
                patch.object(h.Runner, "run") as run, patch.object(h, "MODELS", []), patch.object(h, "TASKS", []):
            out = Path(temp) / "queue.json"
            s.tuning_main(["baseline", "--out", str(out)])
            self.assertEqual(len(json.loads(out.read_text())), 18)
            with self.assertRaises(FileExistsError):
                s.tuning_main(["baseline", "--out", str(out)])
            init.assert_not_called()
            run.assert_not_called()

    def test_unowned_compute_and_nvml_failure_abort(self):
        with patch.object(h.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "")):
            h.require_free_compute()
        with patch.object(h.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, h.GPU1 + ",42,python\n")):
            with self.assertRaisesRegex(h.StopCampaign, "unowned_gpu_workload"):
                h.require_free_compute()
        with patch.object(h.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            with self.assertRaises(subprocess.CalledProcessError):
                h.require_free_compute()

    def test_cosmic_monitor_is_allowed_but_unknown_compute_is_not(self):
        with patch.object(h.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, h.GPU0 + ",42,cosmic-monitor\n")):
            h.require_free_compute()
        with patch.object(h.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, h.GPU0 + ",42,cosmic-app-library\n")):
            h.require_free_compute()
        with patch.object(h.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, h.GPU0 + ",42,unknown-compute\n")):
            with self.assertRaisesRegex(h.StopCampaign, "unowned_gpu_workload"):
                h.require_free_compute()

    def test_tensor_fit_abort_is_a_nonretryable_capacity_result(self):
        self.assertEqual(h.classify_exit(
            "common_fit_params: failed to fit params to free device memory: "
            "llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort"),
            "vram_fit_unavailable")

    def test_effective_context_checked_before_any_request(self):
        with patch.object(p, "api", return_value={"total_slots": 4, "default_generation_settings": {"n_ctx": 32768}}), \
                patch.object(p, "prepared_prompt") as prepare:
            with self.assertRaisesRegex(ValueError, "effective per-slot"):
                p.run_probe("unused", {}, 32700, 2048, 4, 1729, Path("unused"))
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
