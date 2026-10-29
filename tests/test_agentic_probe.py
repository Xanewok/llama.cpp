import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import subprocess

from scripts.instruments import agentic_probe as p
from scripts.instruments import isolated_exec as sandbox


class TrustedGraderTests(unittest.TestCase):
    def test_rust_targets_include_all_trusted_files_but_no_candidate_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, work = root / "sources/exercise", root / "work"
            (source / "tests").mkdir(parents=True)
            (source / "src").mkdir()
            work.mkdir()
            (source / "tests/main.rs").write_text("#[test] fn trusted() {}")
            (source / "tests/extra.rs").write_text("#[test] fn extra() {}")
            (source / "src/lib.rs").write_text("pub fn f() {}")
            task = {"source_dir": "sources/exercise", "language": "rust", "test_paths": ["tests/main.rs"]}
            result = subprocess.CompletedProcess([], 0, "test result: ok. 2 passed; 0 failed; 0 ignored;", "")
            with patch.object(p.isolated_exec, "run", return_value=result) as run:
                grade = p.test_suite(task, work, {}, 30, asset_dir=root)
            self.assertEqual(run.call_args.args[2], ["/rust/bin/cargo", "test", "--offline",
                             "--test", "extra", "--test", "main", "--", "--include-ignored"])
            self.assertEqual(run.call_args.kwargs["readonly"], ["tests/extra.rs", "tests/main.rs"])
            self.assertEqual(grade["passed"], 2)
            self.assertEqual(grade["test_scope"], p.TEST_SCOPE)

    def test_starter_docs_are_not_read_from_the_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "tests/main.rs").write_text("")
            (root / "src/lib.rs").write_text("/// ```compile_fail\n/// trusted();\n/// ```\npub struct X;")
            tests, docs = p.rust_test_plan({"source_dir": ".", "test_paths": ["tests/main.rs"]}, root)
            self.assertEqual(tests, ["tests/main.rs"])
            self.assertEqual(docs, "```compile_fail\ntrusted();\n```")

    def test_enable_all_supplied_javascript_test_forms(self):
        source = "  xtest('a', f);\n  xit('b', f);\n  test.skip('c', f);\ndescribe.skip.each(rows)(f);\nconst s = 'test.skip(';\n"
        enabled = p.enable_js_tests(source)
        self.assertEqual(enabled, "  test('a', f);\n  it('b', f);\n  test('c', f);\ndescribe.each(rows)(f);\nconst s = 'test.skip(';\n")
        self.assertEqual(p.enable_js_tests(enabled), enabled)

    def test_fresh_server_peak_does_not_double_count_output(self):
        self.assertEqual(p.fresh_context_peak({'n_tokens_max': 0}, {'n_tokens_max': 60459.0}, 65536), 60459)
        for before, after in [({'n_tokens_max': 1}, {'n_tokens_max': 2}),
                              ({'n_tokens_max': 0}, {'n_tokens_max': 80805}),
                              ({'n_tokens_max': 0}, {'n_tokens_max': float('nan')}),
                              ({'n_tokens_max': 0}, {})]:
            with self.assertRaises(ValueError):
                p.fresh_context_peak(before, after, 65536)

    def test_cargo_roundtrip_and_only_dependency_changes_enter_grader(self):
        trusted = {"package": {"name": "exercise", "edition": "2021"},
                   "dependencies": {"rand": "0.8"}}
        self.assertEqual(tomllib.loads(p.dump_toml(trusted)), trusted)
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "work"
            grade = Path(temp) / "grade"
            work.mkdir()
            grade.mkdir()
            (work / "Cargo.toml").write_text(p.dump_toml({**trusted, "package": {
                **trusted["package"], "autotests": False}, "lib": {"test": False},
                "dependencies": {"rand": {"version": "0.8", "features": ["std"]}}}))
            (grade / "Cargo.toml").write_text(p.dump_toml(trusted))
            ignored = p.transfer_solution({"solution_path": ["Cargo.toml"]}, work, grade)
            result = tomllib.loads((grade / "Cargo.toml").read_text())
            self.assertEqual(result["package"], trusted["package"])
            self.assertNotIn("lib", result)
            self.assertEqual(result["dependencies"]["rand"]["version"], "0.8")
            self.assertEqual(set(ignored), {"package", "lib"})

    def test_modified_tests_and_package_scripts_are_not_copied(self):
        with tempfile.TemporaryDirectory() as temp:
            work, grade = Path(temp) / "work", Path(temp) / "grade"
            work.mkdir()
            grade.mkdir()
            (work / "solution.js").write_text("implementation")
            (work / "tests.js").write_text("weakened tests")
            (work / "package.json").write_text('{"scripts":{"test":"true"}}')
            (grade / "tests.js").write_text("trusted tests")
            (grade / "package.json").write_text("trusted package")
            p.transfer_solution({"solution_path": ["solution.js"]}, work, grade)
            self.assertEqual((grade / "solution.js").read_text(), "implementation")
            self.assertEqual((grade / "tests.js").read_text(), "trusted tests")
            self.assertEqual((grade / "package.json").read_text(), "trusted package")

    def test_symlinks_and_local_dependency_escapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            work, grade = Path(temp) / "work", Path(temp) / "grade"
            work.mkdir()
            grade.mkdir()
            (work / "solution.js").symlink_to("/etc/passwd")
            with self.assertRaisesRegex(ValueError, "escapes"):
                p.transfer_solution({"solution_path": ["solution.js"]}, work, grade)
            (work / "Cargo.toml").write_text('[dependencies]\nanswer = {path="/oracle"}\n')
            (grade / "Cargo.toml").write_text('[package]\nname="exercise"\n')
            with self.assertRaisesRegex(ValueError, "cached registry"):
                p.transfer_solution({"solution_path": ["Cargo.toml"]}, work, grade)

    def test_client_context_matches_server_without_probe_output_cap(self):
        for context, compact in ((65536, 58000), (102400, 86016)):
            config = tomllib.loads(p.client_config(context))
            self.assertEqual(config["model_context_window"], context)
            self.assertEqual(config["model_auto_compact_token_limit"], compact)
            self.assertLess(compact, context)
            self.assertEqual(config["web_search"], "disabled")
            self.assertNotIn("max_tokens", config)
            self.assertFalse(config["model_providers"]["benchmark"]["requires_openai_auth"])

    def test_outer_sandbox_has_no_host_home_or_network_mount(self):
        assets = {key: "/fake/" + key for key in
                  ("node_root", "rust_root", "cargo_registry", "node_modules", "codex")}
        args = sandbox.sandbox_command("/workdir", assets, ["true"])
        self.assertIn("--unshare-all", args)
        self.assertIn("--clearenv", args)
        self.assertNotIn("--share-net", args)
        self.assertNotIn("/home/xanewok", args)
        self.assertNotIn("/", args)


if __name__ == "__main__":
    unittest.main()
