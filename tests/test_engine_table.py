from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis import engine_table


def row(label: str, gpu: str, decode: float, **extra: object) -> str:
    return json.dumps({"label": label, "gpu": gpu, "decode_tok_s": decode,
                       "completion_tokens": 100, "total_s": 2.0, **extra})


class GroupingTest(unittest.TestCase):
    def rows_from(self, *lines: str) -> dict:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        speed = Path(tmp.name)
        (speed / "engines.jsonl").write_text("\n".join(lines) + "\n")
        with mock.patch.object(engine_table, "SPEED", speed):
            return engine_table.rows()

    def test_one_label_on_two_cards_stays_two_rows(self) -> None:
        # The regression this guards: grouping on the label alone merged a quant
        # benched on two different boxes into one row and let the card read last win,
        # which averaged 70.4 and 40.6 into a 55.5 that described neither.
        data = self.rows_from(
            row("q4|mtp", "NVIDIA GeForce RTX 5070 Ti", 70.4, gpu_count=2),
            row("q4|mtp", "NVIDIA GeForce RTX 5060 Ti", 40.6, gpu_count=2),
        )
        self.assertEqual(len(data), 2)
        self.assertEqual(
            sorted(v["decode_tok_s"][0] for v in data.values()), [40.6, 70.4]
        )

    def test_one_card_at_two_power_caps_stays_two_rows(self) -> None:
        data = self.rows_from(
            row("q4|mtp", "NVIDIA GeForce RTX 3090", 60.1, power_limit_w=390.0),
            row("q4|mtp", "NVIDIA GeForce RTX 3090", 52.0, power_limit_w=250.0),
        )
        self.assertEqual(len(data), 2)

    def test_same_config_and_card_accumulates(self) -> None:
        data = self.rows_from(
            row("q4|mtp", "NVIDIA GeForce RTX 3090", 60.0, power_limit_w=390.0),
            row("q4|mtp", "NVIDIA GeForce RTX 3090", 62.0, power_limit_w=390.0),
        )
        self.assertEqual(len(data), 1)
        self.assertEqual(len(next(iter(data.values()))["decode_tok_s"]), 2)


class CardNameTest(unittest.TestCase):
    def test_canonical_nvidia_name(self) -> None:
        # The old parser took the first word and rendered this as "NVIDIA".
        self.assertEqual(engine_table.card_name("NVIDIA GeForce RTX 3090", 1), "3090")

    def test_keeps_the_suffix(self) -> None:
        self.assertEqual(
            engine_table.card_name("NVIDIA GeForce RTX 5070 Ti", 1), "5070 Ti"
        )

    def test_drops_a_wattage_token_from_a_hand_typed_value(self) -> None:
        self.assertEqual(engine_table.card_name("RTX 3090 390W", 1), "3090")

    def test_counts_multi_gpu(self) -> None:
        self.assertEqual(
            engine_table.card_name("NVIDIA GeForce RTX 5070 Ti", 2), "2x5070 Ti"
        )

    def test_passes_a_bad_value_through_unchanged(self) -> None:
        # 36 historical rows hold a rental label here; normalising would hide them.
        self.assertEqual(
            engine_table.card_name("qwen38-2x5070ti-ttl4", None), "qwen38-2x5070ti-ttl4"
        )

    def test_missing_gpu_is_marked(self) -> None:
        self.assertEqual(engine_table.card_name(None, None), "?")


if __name__ == "__main__":
    unittest.main()
