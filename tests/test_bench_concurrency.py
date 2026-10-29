import json
import unittest

from scripts.instruments.bench_concurrency import read_stream, summarize


def event(value):
    return ("data: " + json.dumps(value) + "\n").encode()


class StreamTests(unittest.TestCase):
    def test_reasoning_is_first_output_and_timings_survive_empty_usage_event(self):
        rows = [event({"choices": [{"delta": {"role": "assistant"}}]}),
                event({"choices": [{"delta": {"reasoning_content": "think"}}]}),
                event({"choices": [{"delta": {}, "finish_reason": "length"}],
                       "timings": {"predicted_n": 5}}),
                event({"choices": [], "usage": {"completion_tokens": 5}}), b"data: [DONE]\n"]
        result = read_stream(rows, 10, now=lambda: 12)
        self.assertEqual(result["ttft_seconds"], 2)
        self.assertEqual(result["timings"]["predicted_n"], 5)
        self.assertEqual(result["usage"]["completion_tokens"], 5)

    def test_missing_usage_and_stream_errors_fail_closed(self):
        for rows in ([b"data: [DONE]\n"], [event({"error": "OOM"})]):
            with self.assertRaises(ValueError):
                read_stream(rows, 0)


class SummaryTests(unittest.TestCase):
    def test_overlapping_rates_use_wall_span_not_summed_request_durations(self):
        rows = [{"started_monotonic": 10, "finished_monotonic": 20,
                 "ttft_seconds": 2, "usage": {"completion_tokens": 100}, "error": None},
                {"started_monotonic": 11, "finished_monotonic": 21,
                 "ttft_seconds": 3, "usage": {"completion_tokens": 100}, "error": None}]
        result = summarize(rows, [{"slots": [{"is_processing": True}, {"is_processing": True}]}])
        self.assertEqual(result["makespan_seconds"], 11)
        self.assertAlmostEqual(result["output_tokens_per_wall_second"], 200 / 11)
        self.assertAlmostEqual(result["output_tokens_per_first_token_to_last_finish_second"], 200 / 9)
        self.assertEqual(result["max_observed_busy_slots"], 2)

    def test_failed_batch_has_no_successful_throughput_number(self):
        result = summarize([{"started_monotonic": 1, "finished_monotonic": 2, "error": "failed"}], [])
        self.assertFalse(result["complete"])
        self.assertIsNone(result["output_tokens_per_wall_second"])


if __name__ == "__main__":
    unittest.main()
