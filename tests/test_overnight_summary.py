import json
from pathlib import Path
import tempfile
import unittest

from scripts.analysis import overnight_summary as report
from scripts.instruments.hillclimb import atomic_json, sealed_json


class CorrectedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / 'campaign-agentic-20260913'
        self.corrected = self.root / 'corrected-bartowski'
        self.original.mkdir()
        self.corrected.mkdir()
        jobs = []
        for task, seconds in [('javascript/one', 100), ('rust/two', 200)]:
            self.save(self.original, 'iq3-control', task, seconds)
            name = self.save(self.original, 'bartowski-iq4-xs', task, None)
            jobs.append({'id': name, 'model': 'bartowski-iq4-xs', 'task': task})
        sealed_json(self.corrected / 'manifest.json', {'queue': jobs})

    def save(self, campaign, model, task, seconds):
        name = 'solve-' + model + '-' + task.replace('/', '-')
        row = {'model': model, 'task': task, 'status': 'failed' if seconds is None else 'passed'}
        if seconds is None:
            row['reason'] = 'probe_failed'
        else:
            row['probe'] = {'complete': True, 'task_success': True, 'timed_out': False,
                            'verified_wall_seconds': seconds, 'prefill_seconds': seconds / 10,
                            'decode_seconds': seconds * .8, 'metrics_delta': {'decode_tokens': 500}}
        sealed_json(campaign / ('job-' + name + '.json'), row)
        return name

    def test_default_keeps_original_failed_arm(self):
        rows = report.coding_rows(self.original)
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(r['status'] == 'failed' for _, r in rows), 2)

    def test_partial_corrected_arm_replaces_old_failures_without_filling_pending(self):
        self.save(self.corrected, 'bartowski-iq4-xs', 'javascript/one', 50)
        summary = report.summarize(self.root, self.corrected)
        row = next(r for r in summary['real_task_totals'] if r['model'] == 'bartowski-iq4-xs')
        self.assertEqual((row['measured_tasks'], row['processed']), (1, 1))
        pair = summary['paired_both_solved_only'][0]
        self.assertEqual(pair['both_solved_pairs'], 1)
        self.assertEqual(pair['candidate_over_baseline_total'], .5)
        text = report.render(summary)
        self.assertIn('pending', text)
        self.assertNotIn('probe_failed', text)

    def test_completed_comparison_is_written_separately(self):
        old = self.root / 'OVERNIGHT-REPORT.md'
        old.write_text('original report')
        self.save(self.corrected, 'bartowski-iq4-xs', 'javascript/one', 50)
        self.save(self.corrected, 'bartowski-iq4-xs', 'rust/two', 100)
        report.write_report(self.root, self.corrected, 'PILOT-COMPARISON')
        self.assertEqual(old.read_text(), 'original report')
        result = json.loads((self.root / 'PILOT-COMPARISON-SUMMARY.json').read_text())
        self.assertEqual(result['paired_both_solved_only'][0]['both_solved_pairs'], 2)
        self.assertEqual(result['paired_both_solved_only'][0]['candidate_over_baseline_total'], .5)

    def test_wrong_model_or_task_set_is_rejected(self):
        self.save(self.corrected, 'unsloth-q4-m', 'javascript/one', 40)
        with self.assertRaisesRegex(ValueError, 'declared Bartowski job'):
            report.coding_rows(self.original, self.corrected)
        sealed_json(self.corrected / 'manifest.json', {'queue': []})
        with self.assertRaisesRegex(ValueError, 'exactly the original'):
            report.coding_rows(self.original, self.corrected)

    def test_completion_watcher_finishes_on_success_or_safety_stop(self):
        for state, expected in [('running', False), ('complete', True), ('stopped', True), ('error', False)]:
            atomic_json(self.corrected / 'state.json', {'status': state})
            self.assertEqual(report.campaign_terminal(self.corrected), expected)


if __name__ == '__main__':
    unittest.main()
