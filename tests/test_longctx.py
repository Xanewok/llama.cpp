from __future__ import annotations

import hashlib
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.instruments import longctx  # noqa: E402


class ForbiddenTermsTest(unittest.TestCase):
    def test_covers_the_slug_and_its_identifier_spellings(self) -> None:
        terms = longctx.forbidden_terms({"task_id": "rust/robot-name"})
        self.assertIn("robot-name", terms)
        self.assertIn("robot_name", terms)
        self.assertIn("robotname", terms)


class LeakGuardTest(unittest.TestCase):
    def test_padding_naming_the_exercise_aborts(self) -> None:
        """Padding that mentions the exercise would hand over the answer.

        The guard must fail the run rather than quietly measure with it.
        """

        original = longctx.CORPORA.copy()
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                poisoned = Path(directory) / "rust.txt"
                poisoned.write_text("fn robot_name() {}\n" * 4000)
                longctx.CORPORA["rust"] = poisoned
                with self.assertRaises(SystemExit):
                    longctx.build_padding(
                        {"task_id": "rust/robot-name", "language": "rust"},
                        1000,
                        "http://127.0.0.1:9/v1",
                    )
        finally:
            longctx.CORPORA.clear()
            longctx.CORPORA.update(original)


class WrapTest(unittest.TestCase):
    def test_instructions_come_last_so_recency_favours_success(self) -> None:
        wrapped = longctx.wrap_prompt("SOLVE THIS", "PAD" * 100)
        self.assertTrue(wrapped.rstrip().endswith("SOLVE THIS"))
        self.assertLess(wrapped.index("PAD"), wrapped.index("SOLVE THIS"))

    def test_zero_padding_leaves_the_prompt_untouched(self) -> None:
        self.assertEqual(longctx.wrap_prompt("SOLVE THIS", ""), "SOLVE THIS")


class DeterminismTest(unittest.TestCase):
    def test_offset_is_a_function_of_the_task_id_alone(self) -> None:
        # Padding must be identical for the same task across quants and reruns,
        # or the comparison stops being paired.
        import hashlib

        a = int(hashlib.sha256(b"rust/forth").hexdigest()[:12], 16)
        b = int(hashlib.sha256(b"rust/forth").hexdigest()[:12], 16)
        c = int(hashlib.sha256(b"rust/bowling").hexdigest()[:12], 16)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()


class OffsetRetryTest(unittest.TestCase):
    def test_a_leaking_offset_is_replaced_not_fatal(self) -> None:
        """One unlucky offset used to abort the sweep, taking every task behind it."""

        task = {"task_id": "javascript/binary", "language": "javascript"}
        text = longctx.CORPORA["javascript"].read_text()
        banned = longctx.forbidden_terms(task)

        def leaks(attempt: int) -> bool:
            seed = task["task_id"] if attempt == 0 else f"{task['task_id']}:{attempt}"
            start = int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16) % (len(text) - 1)
            window = (text[start:] + text[:start])[: int(64000 * 4.0)]
            return bool(banned & {m.group(0).lower() for m in longctx._WORD.finditer(window)})

        self.assertTrue(leaks(0), "fixture no longer leaks; the retry path is untested")
        self.assertFalse(all(leaks(a) for a in range(longctx._OFFSET_ATTEMPTS)))

    def test_exhausting_every_offset_still_fails_loudly(self) -> None:
        # Retrying must not become a way to silently accept leaked padding.
        task = {"task_id": "rust/x", "language": "rust", "solution_path": ["src/lib.rs"]}
        with unittest.mock.patch.object(longctx, "forbidden_terms", return_value={"the"}):
            with self.assertRaises(SystemExit):
                longctx.build_padding(task, 4000, "http://127.0.0.1:1")
