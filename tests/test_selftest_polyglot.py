from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.instruments import run_polyglot
from scripts.corpora import selftest_polyglot


class PolyglotSelftestTest(unittest.TestCase):
    def test_reference_passes_and_stub_pass_is_broken(self) -> None:
        manifest = ROOT / "tests" / "fixtures" / "polyglot_selftest_manifest.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                returncode = selftest_polyglot.main(
                    ["--manifest", str(manifest), "--out", str(report_path), "--timeout", "10"]
                )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            results = {item["task_id"]: item for item in report["tasks"]}
            good = results["fixture-selftest-good"]
            broken = results["fixture-selftest-broken"]
            self.assertEqual(returncode, 1)
            self.assertTrue(good["reference_passes"])
            self.assertTrue(good["stub_fails"])
            self.assertFalse(good["broken"])
            self.assertTrue(broken["reference_passes"])
            self.assertFalse(broken["stub_fails"])
            self.assertTrue(broken["broken"])
            self.assertIn("BROKEN fixture-selftest-broken", stderr.getvalue())
            with self.assertRaisesRegex(SystemExit, "fixture-selftest-broken"):
                run_polyglot.validate_selftest_report(report_path, run_polyglot.load_manifest(manifest))

    def test_verified_report_marks_emitted_row(self) -> None:
        source = ROOT / "tests" / "fixtures" / "polyglot" / "selftest_good"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            manifest = temp / "manifest.json"
            report = temp / "report.json"
            output = temp / "results.jsonl"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "fixture-selftest-good",
                            "language": "python",
                            "source_dir": str(source),
                            "test_cmd": ["python", "-m", "unittest", "-q"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                selftest_polyglot.main(
                    ["--manifest", str(manifest), "--out", str(report), "--timeout", "10"]
                ),
                0,
            )
            self.assertEqual(
                run_polyglot.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--selftest-report",
                        str(report),
                        "--lane",
                        "qwen",
                        "--quant",
                        "fixture",
                        "--run-index",
                        "0",
                        "--out",
                        str(output),
                        "--dry-run",
                    ]
                ),
                0,
            )
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(row["harness_verified"])

    def test_language_specific_test_activation(self) -> None:
        self.assertEqual(
            run_polyglot.task_test_command(
                {"language": "rust", "test_cmd": ["cargo", "test", "--quiet"]}
            ),
            ["cargo", "test", "--quiet", "--", "--include-ignored"],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".meta").mkdir()
            (root / ".meta" / "config.json").write_text(
                json.dumps({"files": {"test": ["exercise.spec.js"]}}), encoding="utf-8"
            )
            test_path = root / "exercise.spec.js"
            source_path = root / "exercise.js"
            test_path.write_text("xtest('disabled', () => {});\n", encoding="utf-8")
            source_path.write_text("xtest('not a test file', () => {});\n", encoding="utf-8")

            replacements = run_polyglot.prepare_test_harness(
                {"language": "javascript"}, root
            )

            self.assertEqual(replacements, 1)
            self.assertIn("test('disabled'", test_path.read_text(encoding="utf-8"))
            self.assertIn("xtest('not a test file'", source_path.read_text(encoding="utf-8"))

    def test_reference_overlay_ignores_unmatched_auxiliary_solution_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".meta").mkdir()
            (root / "src").mkdir()
            (root / ".meta" / "example.rs").write_text("reference\n")
            (root / "src" / "lib.rs").write_text("stub\n")
            (root / "Cargo.toml").write_text("[package]\nname='fixture'\n")
            selftest_polyglot._overlay_reference(
                {
                    "example_path": [".meta/example.rs"],
                    "solution_path": ["src/lib.rs", "Cargo.toml"],
                },
                root,
            )
            self.assertEqual((root / "src" / "lib.rs").read_text(), "reference\n")
            self.assertIn("name='fixture'", (root / "Cargo.toml").read_text())


if __name__ == "__main__":
    unittest.main()
