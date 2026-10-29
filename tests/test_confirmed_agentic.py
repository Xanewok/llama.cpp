import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import ROOT
from scripts.instruments import agentic_campaign as a, confirmed_agentic as c, hillclimb as h


def observations(candidate_tg=32):
    rows = []
    for model in c.MODELS[1:]:
        for task in c.CONFIRM_TASKS:
            for seed in c.SEEDS:
                for ubatch in (128, 512):
                    rows.append({'model': model, 'task': task, 'seed': seed, 'status': 'passed',
                                 'config': {'context': 102400, 'slots': 1, 'mode': 'tensor',
                                            'split': '69,31', 'maximum': 3, 'ubatch': ubatch},
                                 'prefill_tps': 400 if ubatch == 128 else 450,
                                 'decode_tps': 30 if ubatch == 128 else candidate_tg,
                                 'probe': dict.fromkeys(['complete', 'mtp_live', 'depth_verified', 'overlap_verified'], True)})
    return rows


class ConfirmationTests(unittest.TestCase):
    def test_improvement_selects_candidate_but_does_not_promote_production(self):
        selected = c.choose_microbatches(observations(36))
        for value in selected.values():
            self.assertEqual(value['ubatch'], 512)
            self.assertTrue(value['near_100k_speed_gate'])
            self.assertFalse(value['production_qualified'])
            self.assertEqual(value['valid_repeats'], {'128': 12, '512': 12})

    def test_decode_regression_keeps_baseline_despite_better_prefill(self):
        self.assertTrue(all(v['ubatch'] == 128 for v in c.choose_microbatches(observations(25)).values()))

    def test_memory_failure_or_idle_mtp_is_not_a_valid_confirmation(self):
        for change in ({'status': 'failed'}, {'probe': {'complete': True, 'mtp_live': False}}):
            rows = observations()
            rows[1].update(change)
            selected = c.choose_microbatches(rows)[c.MODELS[1]]
            self.assertEqual(selected['ubatch'], 128)
            self.assertEqual(selected['valid_repeats']['512'], 11)

    def test_no_safe_observations_excludes_model(self):
        rows = observations()
        for row in rows:
            row['status'] = 'failed'
        self.assertTrue(all(v['ubatch'] is None for v in c.choose_microbatches(rows).values()))

    def test_missing_duplicate_and_changed_context_fail_closed(self):
        rows = observations()
        invalid = [rows[:-1], rows + [rows[0]]]
        altered = copy.deepcopy(rows)
        altered[0]['config']['context'] = 65536
        invalid.append(altered)
        for sample in invalid:
            with self.assertRaises(ValueError):
                c.choose_microbatches(sample)

    def test_full_suite_covers_all_tasks_and_only_declared_variants(self):
        tasks = [t['task_id'] for t in json.loads((ROOT / 'subsets/polyglot.json').read_text())]
        jobs = h.validate_profiles(c.suite_queue(tasks), allowed_tasks=set(tasks))
        self.assertEqual(len(jobs), 234)
        self.assertEqual(len({j['id'] for j in jobs}), 234)
        for model in c.MODELS:
            arm = [j for j in jobs if j['model'] == model]
            self.assertEqual({j['task'] for j in arm}, set(tasks))
            for job in arm:
                self.assertEqual({v['slots'] for v in job['configs']}, {1})
                self.assertEqual({v['ubatch'] for v in job['configs']}, {512} if model == 'iq3-control' else {128, 512})

    def test_runtime_resolves_only_predeclared_configuration(self):
        runner = c.ConfirmedRunner.__new__(c.ConfirmedRunner)
        runner.selection = {'models': {c.MODELS[1]: {'ubatch': 512}}}
        job = c.suite_queue(['javascript/affine-cipher'])[1]
        original = copy.deepcopy(job)
        with patch.object(a.AgenticRunner, 'execute_job', return_value={'status': 'passed'}) as execute:
            runner.execute_job(job)
        self.assertEqual(execute.call_args.args[0]['configs'][0]['ubatch'], 512)
        self.assertEqual(job, original)
        runner.selection['models'][c.MODELS[1]]['ubatch'] = 256
        with self.assertRaises(ValueError):
            runner.execute_job(job)

    def test_model_first_is_a_stable_permutation_with_unchanged_jobs(self):
        tasks = ['javascript/affine-cipher', 'rust/forth', 'javascript/bowling']
        old, new = c.suite_queue(tasks), c.suite_queue(tasks, 'model-first')
        self.assertEqual({j['id']: j for j in old}, {j['id']: j for j in new})
        self.assertEqual([j['model'] for j in new], [m for m in c.MODELS for _ in tasks])
        for model in c.MODELS:
            self.assertEqual([j['task'] for j in new if j['model'] == model], tasks)
        with self.assertRaises(ValueError):
            c.suite_queue(tasks, 'unknown')

    def test_predecessor_error_can_recover_without_losing_continuation(self):
        with tempfile.TemporaryDirectory() as temp:
            campaign, previous = Path(temp) / 'next', Path(temp) / 'previous'
            campaign.mkdir(); previous.mkdir()
            h.atomic_json(previous / 'state.json', {'status': 'error', 'completed': {}})
            def recover(_):
                h.atomic_json(previous / 'state.json', {'status': 'complete', 'completed': {}})
            with patch.object(a.time, 'sleep', side_effect=recover):
                self.assertTrue(a.wait_for_predecessor(campaign, previous, time.time() + 60))

    def test_persistent_predecessor_error_blocks_instead_of_waiting_all_day(self):
        with tempfile.TemporaryDirectory() as temp:
            campaign, previous = Path(temp) / 'next', Path(temp) / 'previous'
            campaign.mkdir(); previous.mkdir()
            h.atomic_json(previous / 'state.json', {'status': 'error', 'completed': {}})
            with patch.object(a.time, 'monotonic', side_effect=[0, 0, 181]), patch.object(a.time, 'sleep'):
                self.assertFalse(a.wait_for_predecessor(campaign, previous, time.time() + 1000))
            self.assertEqual(json.loads((campaign / 'stage-status.json').read_text())['phase'], 'blocked')

    def test_instrument_failure_stops_but_failed_task_is_still_measured(self):
        runner = c.ConfirmedRunner.__new__(c.ConfirmedRunner)
        job = c.suite_queue(['javascript/affine-cipher'])[0]
        for reason in ('probe_failed', 'probe_incomplete'):
            with patch.object(a.AgenticRunner, 'execute_job', return_value={'status': 'failed', 'reason': reason}):
                with self.assertRaises(h.StopCampaign):
                    runner.execute_job(job)
        row = {'status': 'passed', 'probe': {'task_success': False}}
        with patch.object(a.AgenticRunner, 'execute_job', return_value=row):
            self.assertEqual(runner.execute_job(job), row)

    def test_handoff_lock_race_retries_without_touching_the_other_server(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = c.ConfirmedRunner.__new__(c.ConfirmedRunner)
            runner.path = Path(temp)
            runner.deadline = time.monotonic() + 300
            runner.lockfiles = []
            with patch.object(h.Runner, 'acquire', side_effect=[h.CampaignBusy('busy'), None]) as acquire, \
                    patch.object(c.time, 'sleep'):
                runner.acquire()
            self.assertEqual(acquire.call_count, 2)

    def test_fit_rejection_maps_to_declared_case_not_placeholder_prompt(self):
        row = observations()[1]
        job = {'id': 'one', 'task': row['task'], 'model': row['model'], 'seed': row['seed']}
        rejected = {**row, 'job': 'one-fit', 'target': 0, 'task': 'placeholder', 'seed': 1729,
                    'status': 'failed', 'reason': 'vram_reserve'}
        mapped = c.confirmation_observation(job, rejected)
        self.assertEqual((mapped['task'], mapped['seed']), (job['task'], job['seed']))
        self.assertEqual(rejected['task'], 'placeholder')
        rows = observations()
        rows[1] = mapped
        self.assertEqual(c.choose_microbatches(rows)[job['model']]['valid_repeats']['512'], 11)
        with self.assertRaises(ValueError):
            c.confirmation_observation(job, {**rejected, 'status': 'passed'})

    def test_selection_is_persisted_and_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            source, campaign = Path(temp) / 'source', Path(temp) / 'campaign'
            source.mkdir(); campaign.mkdir()
            jobs = []
            for n, row in enumerate(observations()):
                job = {'id': str(n), 'task': row['task'], 'model': row['model'], 'seed': row['seed']}
                jobs.append(job)
                h.sealed_json(source / f'job-{n}.json', {**row, 'job': str(n), 'target': 98304})
            h.sealed_json(source / 'manifest.json', {'queue': jobs})
            selected = c.resolve_selection(source, campaign)
            self.assertEqual(c.resolve_selection(source, campaign), selected)
            changed = h.read_sealed(source / 'job-0.json')
            changed['prefill_tps'] += 1
            h.sealed_json(source / 'job-0.json', changed)
            with self.assertRaisesRegex(ValueError, 'changed after freezing'):
                c.resolve_selection(source, campaign)


if __name__ == '__main__':
    unittest.main()
