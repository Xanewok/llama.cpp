from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.instruments import emit  # noqa: E402


BEFORE = {
    "prompt_tokens_total": 1000.0,
    "prompt_tokens_cached_total": 400.0,
    "prompt_seconds_total": 10.0,
    "tokens_predicted_total": 500.0,
    "tokens_predicted_seconds_total": 25.0,
    "n_tokens_max": 900.0,
}
AFTER = {
    "prompt_tokens_total": 3000.0,
    "prompt_tokens_cached_total": 1800.0,
    "prompt_seconds_total": 11.0,
    "tokens_predicted_total": 1100.0,
    "tokens_predicted_seconds_total": 35.0,
    "n_tokens_max": 2400.0,
}


class MetricsDeltaTest(unittest.TestCase):
    def test_delta_is_this_tasks_share(self) -> None:
        d = emit.metrics_delta(BEFORE, AFTER)
        self.assertEqual(d["prefill_tokens"], 2000.0)
        self.assertEqual(d["cached_prompt_tokens"], 1400.0)
        self.assertEqual(d["decode_tokens"], 600.0)
        self.assertEqual(d["prefill_tps"], 2000.0)
        self.assertEqual(d["decode_tps"], 60.0)
        # n_tokens_max is process-wide, so it is recorded as such; per-task depth
        # comes from ContextSampler instead.
        self.assertEqual(d["server_high_water_tokens"], 2400.0)
        self.assertNotIn("peak_context_tokens", d)

    def test_counter_reset_discards_the_reading(self) -> None:
        # A server restart between scrapes rewinds the counters; clamping to zero
        # would publish a plausible wrong number instead of an honest absence.
        self.assertEqual(emit.metrics_delta(AFTER, BEFORE), {})

    def test_a_failed_scrape_yields_nothing(self) -> None:
        self.assertEqual(emit.metrics_delta({}, AFTER), {})
        self.assertEqual(emit.metrics_delta(BEFORE, {}), {})

    def test_zero_elapsed_does_not_divide(self) -> None:
        same = dict(BEFORE)
        d = emit.metrics_delta(BEFORE, same)
        self.assertNotIn("prefill_tps", d)
        self.assertNotIn("decode_tps", d)


class ContextSamplerTest(unittest.TestCase):
    def _slot(self, prompt: int, decoded: int, processing: bool = True) -> list:
        return [{
            "is_processing": processing,
            "n_prompt_tokens": prompt,
            "next_token": [{"n_decoded": decoded}],
        }]

    def test_depth_is_prompt_plus_decoded(self) -> None:
        sampler = emit.ContextSampler("http://x/v1")
        sampler._sample = lambda: sampler.__dict__.__setitem__(
            "peak", max(sampler.peak, 72382 + 53382)
        )
        sampler._sample()
        self.assertEqual(sampler.stop(), 125764)

    def test_idle_slots_do_not_contribute(self) -> None:
        sampler = emit.ContextSampler("http://x/v1")
        slots = self._slot(9999, 9999, processing=False)
        for slot in slots:
            if slot.get("is_processing"):
                self.fail("idle slot treated as active")
        self.assertEqual(sampler.stop(), 0)

    def test_sampling_failure_never_fails_a_task(self) -> None:
        # Observational only: an unreachable server must cost the reading, not the run.
        sampler = emit.ContextSampler("http://127.0.0.1:9/v1", interval=0.01)
        sampler.start()
        sampler.join(0.2)
        self.assertEqual(sampler.stop(), 0)


class QuantGuardTest(unittest.TestCase):
    def _ctx(self, quant: str) -> emit.RunContext:
        return emit.RunContext(out="unused", lane="quant", quant=quant)

    def test_mismatch_between_label_and_served_file_aborts(self) -> None:
        original = emit.probe_server
        emit.probe_server = lambda url, timeout=10.0: {
            "served_model_path": "/m/Qwen3.8-27B-UD-IQ3_XXS.gguf"
        }
        try:
            with self.assertRaises(SystemExit):
                emit.apply_server_probe(self._ctx("UD-Q6_K_M"), "http://x/v1")
            ctx = emit.apply_server_probe(self._ctx("UD-IQ3_XXS"), "http://x/v1")
            self.assertIn("IQ3_XXS", ctx.served_model_path)
        finally:
            emit.probe_server = original

    def test_unreachable_server_aborts_rather_than_recording_blind(self) -> None:
        original = emit.probe_server
        emit.probe_server = lambda url, timeout=10.0: {}
        try:
            with self.assertRaises(SystemExit):
                emit.apply_server_probe(self._ctx("UD-Q6_K_M"), "http://x/v1")
        finally:
            emit.probe_server = original


class CodexUsageTest(unittest.TestCase):
    def _stream(self, usage: dict) -> str:
        import json as _json
        return _json.dumps({"type": "turn.completed", "usage": usage})

    def test_reads_both_spellings_of_cached_input_tokens(self) -> None:
        flat = emit.parse_codex_stats(
            self._stream({"input_tokens": 100, "output_tokens": 10,
                          "cached_input_tokens": 60})
        )
        nested = emit.parse_codex_stats(
            self._stream({"input_tokens": 100, "output_tokens": 10,
                          "input_tokens_details": {"cached_tokens": 60}})
        )
        self.assertEqual(flat.cached_input_tokens, 60)
        self.assertEqual(nested.cached_input_tokens, 60)

    def test_absent_cache_reporting_stays_null_not_zero(self) -> None:
        # Null means "the model did not report it"; zero would claim a measured miss.
        stats = emit.parse_codex_stats(
            self._stream({"input_tokens": 100, "output_tokens": 10})
        )
        self.assertIsNone(stats.cached_input_tokens)
        self.assertIsNone(stats.reasoning_tokens)

    def test_reasoning_tokens_are_summed_when_reported(self) -> None:
        stats = emit.parse_codex_stats(
            self._stream({"input_tokens": 1, "output_tokens": 900,
                          "output_tokens_details": {"reasoning_tokens": 850}})
        )
        self.assertEqual(stats.reasoning_tokens, 850)


class SchemaTest(unittest.TestCase):
    def test_every_context_field_is_in_the_row_schema(self) -> None:
        missing = emit._CONTEXT_FIELDS - set(emit.RESULT_FIELDS)
        self.assertFalse(missing, f"context fields absent from rows: {sorted(missing)}")

    def test_row_schema_has_no_duplicates(self) -> None:
        self.assertEqual(len(emit.RESULT_FIELDS), len(set(emit.RESULT_FIELDS)))


if __name__ == "__main__":
    unittest.main()


class MetricsSchemaTest(unittest.TestCase):
    def test_every_metric_delta_key_is_a_result_field(self) -> None:
        """metrics_delta and RESULT_FIELDS drift apart silently: adding a counter
        makes build_row reject the row, which kills the leg at its first emit --
        after the task has already been paid for."""

        before = {name: 0.0 for name in emit._METRIC_NAMES.values()}
        after = {name: 1.0 for name in emit._METRIC_NAMES.values()}
        produced = set(emit.metrics_delta(before, after))
        self.assertTrue(produced, "delta produced nothing; the check would be vacuous")
        self.assertFalse(
            produced - set(emit.RESULT_FIELDS),
            f"metrics_delta emits fields build_row rejects: {sorted(produced - set(emit.RESULT_FIELDS))}",
        )
