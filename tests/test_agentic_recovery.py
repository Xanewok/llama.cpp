import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
import subprocess
from unittest.mock import patch

from scripts.instruments import agentic_campaign as a, confirmed_agentic as c, hillclimb as h
from tests.test_hillclimb import safe_sample


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        for name in ('snapshot', 'attempts', 'artifacts'):
            (self.path / name).mkdir()
        h.sealed_json(self.path / 'manifest.json', {'port': 8081, 'queue': [], 'max_attempts': 3})
        h.atomic_json(self.path / 'state.json', {'completed': {}, 'status': 'prepared', 'owned': []})
        self.runner = c.ConfirmedRunner(self.path, time.time() + 3600, {'models': {}})
        self.runner.last_report = time.monotonic()

    def failure(self, finished=1000):
        attempt = self.path / 'attempts/pressure-01'
        attempt.mkdir(exist_ok=True)
        before = safe_sample()
        before['gpus'][h.GPU0]['memory.free'] = 15000
        (attempt / 'telemetry.jsonl').write_text(json.dumps(before) + '\n')
        return {'reason': 'vram_reserve: ' + h.GPU0, 'config': {'devices': [h.GPU0]},
                'attempt': 'attempts/pressure-01', 'finished_at': finished,
                'minimum_free_mib': {h.GPU0: 400}}

    def test_actual_timeout_signature_stops_without_retry_despite_warning(self):
        def reject(_):
            (self.runner.current_dir / 'server.log').write_text(
                "unsupported Responses tool type 'namespace' skipped\n"
                "CUDA error: the launch timed out and was terminated\n")
            raise h.RejectArm('probe_failed')
        with patch.object(h, 'endpoint_free', side_effect=reject) as launch, \
                patch.object(self.runner, 'poll'), patch.object(self.runner, 'event'):
            with self.assertRaisesRegex(h.StopCampaign, 'gpu_fault'):
                self.runner.attempt('fault', {'id': 'test'}, {}, 0)
        self.assertEqual(launch.call_count, 1)
        result = h.read_sealed(next((self.path / 'attempts').glob('*/result.json')))
        self.assertEqual(result['reason'], 'gpu_fault')

    def test_pressure_retries_are_bounded_and_persist_across_resume(self):
        with patch.object(h, 'endpoint_free', side_effect=h.RejectArm('vram_reserve: ' + h.GPU0)) as launch, \
                patch.object(self.runner, 'poll'), patch.object(self.runner, 'sleep'), \
                patch.object(self.runner, 'event'), patch.object(self.runner, 'wait_before_retry') as wait:
            result = self.runner.attempt('pressure', {'id': 'test'}, {}, 0)
            self.assertEqual(launch.call_count, 3)
            self.assertEqual(wait.call_count, 2)
            resumed = c.ConfirmedRunner(self.path, time.time() + 3600, {'models': {}})
            self.assertEqual(resumed.attempt('pressure', {'id': 'test'}, {}, 0), result)
            self.assertEqual(launch.call_count, 3)

    def test_exhausted_pressure_keeps_job_pending_and_does_not_skip_next_tasks(self):
        jobs = c.suite_queue(['javascript/affine-cipher'])
        self.runner.manifest['queue'] = jobs
        row = {'status': 'skipped', 'reason': 'no_safe_model_fit', 'fit_reason': 'oom'}
        with patch.object(self.runner, 'acquire'), patch.object(self.runner, 'poll'), \
                patch.object(self.runner, 'report'), patch.object(self.runner, 'event'), \
                patch.object(a.AgenticRunner, 'execute_job', return_value=row) as execute:
            self.runner.run()
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(self.runner.state['status'], 'stopped')
        self.assertEqual(self.runner.state['completed'], {})
        self.assertFalse(list(self.path.glob('job-*.json')))

    def test_wait_requires_stable_headroom_and_preserves_deadline(self):
        result = self.failure()
        clock = [1005.0]
        def poll(reserves=True):
            self.assertFalse(reserves)
            sample = safe_sample()
            sample['gpus'][h.GPU0]['memory.free'] = 15300 if clock[0] >= 1008 else 15100
            self.runner.samples.append(sample)
        def sleep(seconds):
            clock[0] += seconds
        with patch.object(c.time, 'time', side_effect=lambda: clock[0]), \
                patch.object(c.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(c.time, 'sleep', side_effect=sleep), patch.object(self.runner, 'poll', side_effect=poll), \
                patch.object(self.runner, 'event') as event, patch.object(self.runner, 'report'):
            self.runner.wait_before_retry('one', result, 5)
        self.assertGreaterEqual(clock[0], 1018)
        started = next(x for x in event.call_args_list if x.args[0] == 'memory_wait')
        self.assertEqual(started.kwargs['required_free_mib'], {h.GPU0: 15240})
        self.assertEqual(started.kwargs['until_epoch'], 1900)
        self.assertNotIn('resource_wait', self.runner.state)

    def test_expired_recovery_is_not_extended_by_restart(self):
        result = self.failure(time.time() - 901)
        with patch.object(self.runner, 'poll'), patch.object(self.runner, 'event'), patch.object(self.runner, 'report'):
            with self.assertRaisesRegex(h.StopCampaign, 'memory_pressure_unresolved'):
                self.runner.wait_before_retry('one', result, 0)

    def test_wait_still_honors_stop_deadline_and_thermal_limits(self):
        result = self.failure(time.time())
        for condition in ('stop', 'deadline', 'thermal'):
            with self.subTest(condition=condition):
                self.runner.deadline = 0 if condition == 'deadline' else time.monotonic() + 100
                self.runner.stopping = condition == 'stop'
                self.runner.last_sample = 0
                sample = safe_sample()
                if condition == 'thermal':
                    sample['gpus'][h.GPU0]['temperature.gpu'] = 80
                with patch.object(h, 'telemetry', return_value=sample), patch.object(self.runner, 'event'), \
                        patch.object(self.runner, 'report'):
                    with self.assertRaises(h.StopCampaign):
                        self.runner.wait_before_retry('one', result, 0)

    def test_memory_wait_refuses_live_children(self):
        self.runner.children = [object()]
        with self.assertRaisesRegex(h.StopCampaign, 'stopped_children'):
            self.runner.wait_before_retry('one', {'reason': 'oom'}, 0)


class WatchdogRecoveryTests(unittest.TestCase):
    failure = RecoveryTests.failure

    def setUp(self):
        RecoveryTests.setUp(self)
        self.runner.manifest['continuation'] = {'watchdog_recovery': c.WATCHDOG_RECOVERY}
        h.sealed_json(self.path / 'manifest.json', self.runner.manifest)
        self.runner.watchdog_policy = c.WATCHDOG_RECOVERY

    def test_only_the_observed_timeout_signature_is_retryable(self):
        timeout = 'CUDA error: the launch timed out and was terminated\n'
        self.assertTrue(self.runner.retryable_gpu_fault(timeout))
        self.assertTrue(self.runner.retryable_gpu_fault(timeout + 'NVRM: Xid (PCI:0000:08:00): 8, pid=42'))
        for text in ('illegal memory access', 'device-side assert', 'unspecified launch failure',
                     'fallen off the bus', 'NVRM: Xid (PCI:0000:08:00): 79, pid=42'):
            self.assertFalse(self.runner.retryable_gpu_fault(timeout + text))
        self.assertFalse(self.runner.retryable_gpu_fault('NVRM: Xid: 8'))
        self.assertFalse(h.Runner.retryable_gpu_fault(self.runner, timeout))

    def test_kernel_check_requires_xid8_and_rejects_other_or_unreadable_faults(self):
        good = 'NVRM: Xid (PCI:0000:08:00): 8, pid=42\n'
        for output, code, accepted in ((good, 0, True), (good + 'NVRM: Xid: 31', 0, False),
                                       ('', 1, False), (good, 2, False)):
            with self.subTest(output=output, code=code), patch.object(c.subprocess, 'run',
                    return_value=subprocess.CompletedProcess([], code, output, '')):
                if accepted:
                    self.assertEqual(c.watchdog_kernel_check(1000), good)
                else:
                    with self.assertRaisesRegex(h.StopCampaign, 'kernel_health'):
                        c.watchdog_kernel_check(1000)
        with patch.object(c.subprocess, 'run', side_effect=subprocess.TimeoutExpired('journalctl', 10)):
            with self.assertRaisesRegex(h.StopCampaign, 'kernel_health'):
                c.watchdog_kernel_check(1000)

    def test_three_retries_are_persistent_and_exhaustion_keeps_job_pending(self):
        def reject(_):
            (self.runner.current_dir / 'server.log').write_text(
                'unsupported Responses tool type namespace skipped\n'
                'CUDA error: the launch timed out and was terminated\n')
            raise h.RejectArm('probe_failed')
        with patch.object(h, 'endpoint_free', side_effect=reject) as launch, \
                patch.object(self.runner, 'poll'), patch.object(self.runner, 'sleep'), \
                patch.object(self.runner, 'event'), patch.object(self.runner, 'wait_before_retry') as wait:
            result = self.runner.attempt('watchdog', {'id': 'test'}, {}, 0)
        self.assertEqual(result['reason'], 'cuda_watchdog')
        self.assertEqual(launch.call_count, 4)
        self.assertEqual(wait.call_count, 3)
        self.assertEqual([self.runner.retry_backoff([result] * n, n) for n in (1, 2, 3)], [20, 40, 60])
        resumed = c.ConfirmedRunner(self.path, time.time() + 3600, {'models': {}})
        with patch.object(h, 'endpoint_free') as launch:
            self.assertEqual(resumed.attempt('watchdog', {'id': 'test'}, {}, 0), result)
            launch.assert_not_called()
        with patch.object(a.AgenticRunner, 'execute_job', return_value=result):
            with self.assertRaisesRegex(h.StopCampaign, 'watchdog_retries_exhausted'):
                resumed.checked_execute({'id': 'watchdog'})
        self.assertEqual(resumed.state['completed'], {})

    def test_partial_retry_does_not_reset_the_retry_budget(self):
        failure = {'reason': 'cuda_watchdog'}
        self.assertEqual(self.runner.retry_count([failure], 4), 4)
        self.assertGreaterEqual(self.runner.retry_count([failure], 4), self.runner.retry_limit('cuda_watchdog'))
        self.assertEqual(self.runner.retry_count([{'reason': 'oom'}], 4), 1)

    def test_success_after_watchdog_uses_a_new_attempt_and_is_not_repeated(self):
        self.runner.manifest['binary'] = '/fake/server'
        model = {'id': 'test', 'path': '/fake/model', 'sha256': 'test'}
        config = {'devices': [h.GPU0], 'context': 65536, 'slots': 1, 'maximum': 3,
                  'mode': 'layer', 'split': '1', 'ubatch': 512, 'workload': 'agentic'}
        visited = []
        def endpoint(_):
            visited.append(self.runner.current_dir)
            if len(visited) == 1:
                (self.runner.current_dir / 'server.log').write_text('CUDA error: the launch timed out and was terminated')
                raise h.RejectArm('gpu_fault')
        def api(url, path, **kwargs):
            return {'status': 'ok'} if path == '/health' else {
                'total_slots': 1, 'default_generation_settings': {'n_ctx': 65536}, 'model_path': '/fake/model'}
        def run_probe(config, target, task, seed, out):
            h.atomic_json(out / 'probe.json', {'complete': True, 'mtp_live': True, 'depth_verified': True,
                'overlap_verified': True, 'task_success': True, 'metrics_delta': {'decode_tps': 42}})
            return 0
        from unittest.mock import Mock
        server = Mock()
        server.poll.return_value = None
        with patch.object(h, 'endpoint_free', side_effect=endpoint) as launch, \
                patch.object(h, 'api', side_effect=api), patch.object(self.runner, 'spawn', return_value=server), \
                patch.object(self.runner, 'poll'), patch.object(self.runner, 'sleep'), patch.object(self.runner, 'event'), \
                patch.object(self.runner, 'wait_before_retry'), patch.object(self.runner, 'run_probe', side_effect=run_probe):
            result = self.runner.attempt('watchdog', model, config, 0)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(launch.call_count, 2)
            self.assertNotEqual(visited[0], visited[1])
            self.assertEqual(h.read_sealed(visited[0] / 'result.json')['reason'], 'cuda_watchdog')
            self.assertEqual(self.runner.attempt('watchdog', model, config, 0), result)
            self.assertEqual(launch.call_count, 2)

    def test_a_different_gpu_fault_stays_fatal_even_on_resume(self):
        def reject(_):
            (self.runner.current_dir / 'server.log').write_text('CUDA illegal memory access')
            raise h.RejectArm('probe_failed')
        with patch.object(h, 'endpoint_free', side_effect=reject), \
                patch.object(self.runner, 'poll'), patch.object(self.runner, 'event'):
            with self.assertRaisesRegex(h.StopCampaign, 'gpu_fault'):
                self.runner.attempt('fatal', {'id': 'test'}, {}, 0)
        with patch.object(h, 'endpoint_free') as launch:
            with self.assertRaisesRegex(h.StopCampaign, 'gpu_fault'):
                self.runner.attempt('fatal', {'id': 'test'}, {}, 0)
            launch.assert_not_called()

    def watchdog_failure(self, finished):
        result = self.failure(finished)
        result.update(reason='cuda_watchdog', elapsed_seconds=400)
        return result

    def test_recovery_waits_for_stable_released_memory_and_backoff(self):
        result = self.watchdog_failure(1000)
        clock = [1005.0]
        def poll(reserves=True):
            self.assertFalse(reserves)
            sample = safe_sample()
            sample['gpus'][h.GPU0]['memory.free'] = 15300 if clock[0] >= 1018 else 15100
            self.runner.samples.append(sample)
        with patch.object(c.time, 'time', side_effect=lambda: clock[0]), \
                patch.object(c.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(c.time, 'sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)), \
                patch.object(self.runner, 'poll', side_effect=poll), patch.object(self.runner, 'event') as event, \
                patch.object(self.runner, 'report'), patch.object(c, 'watchdog_kernel_check', return_value='Xid8') as kernel, \
                patch.object(h, 'endpoint_free') as endpoint:
            self.runner.wait_before_retry('watchdog', result, 15)
        self.assertEqual(clock[0], 1028)
        self.assertEqual(kernel.call_count, 2)
        endpoint.assert_called_once()
        waiting = next(call for call in event.call_args_list if call.args[0] == 'watchdog_wait')
        self.assertEqual(waiting.kwargs['until_epoch'], 1120)
        self.assertNotIn('resource_wait', self.runner.state)

    def test_expired_health_window_and_live_children_block_restart(self):
        with patch.object(c, 'watchdog_kernel_check', return_value='Xid8'), \
                patch.object(self.runner, 'poll'), patch.object(self.runner, 'event'), patch.object(self.runner, 'report'):
            with self.assertRaisesRegex(h.StopCampaign, 'watchdog_health_unresolved'):
                self.runner.wait_before_retry('watchdog', self.watchdog_failure(time.time() - 121), 0)
        self.runner.state['owned'] = [{'pid': 1}]
        with self.assertRaisesRegex(h.StopCampaign, 'stopped_children'):
            self.runner.wait_before_retry('watchdog', self.watchdog_failure(time.time()), 0)

    def test_wait_honors_stop_deadline_temperature_power_and_ram_guards(self):
        for condition in ('stop', 'deadline', 'thermal', 'power', 'ram'):
            with self.subTest(condition=condition):
                self.runner.deadline = 0 if condition == 'deadline' else time.monotonic() + 100
                self.runner.stopping = condition == 'stop'
                self.runner.last_sample = 0
                sample = safe_sample()
                if condition == 'thermal':
                    sample['gpus'][h.GPU0]['temperature.gpu'] = 80
                if condition == 'power':
                    sample['gpus'][h.GPU0]['power.limit'] = 180
                if condition == 'ram':
                    sample['ram_available_mib'] = 2047
                with patch.object(h, 'telemetry', return_value=sample), \
                        patch.object(c, 'watchdog_kernel_check', return_value='Xid8'), \
                        patch.object(self.runner, 'event'), patch.object(self.runner, 'report'):
                    with self.assertRaises(h.StopCampaign):
                        self.runner.wait_before_retry('watchdog', self.watchdog_failure(time.time()), 0)


class MigrationTests(unittest.TestCase):
    def fixture(self, root):
        source, target = root / 'source', root / 'target'
        for path in (source, target):
            (path / 'attempts').mkdir(parents=True)
        jobs = [{'id': i, 'model': 'iq3-control', 'task': 'javascript/' + i} for i in ('done', 'failed', 'pending')]
        manifest = {k: k for k in ('engine_commit', 'binary', 'binary_pins', 'devices', 'port', 'models', 'safety')}
        manifest.update(queue=jobs, max_attempts=3, continuation={'confirmation': 'same'},
                        snapshot_pins={'scripts/instruments/hillclimb.py': 'old', 'grader': 'unchanged'})
        h.sealed_json(source / 'manifest.json', manifest)
        manifest = copy.deepcopy(manifest)
        manifest['snapshot_pins']['scripts/instruments/hillclimb.py'] = 'new'
        manifest['continuation']['memory_recovery'] = c.MEMORY_RECOVERY
        h.sealed_json(target / 'manifest.json', manifest)
        (source / 'attempts/abc-01').mkdir()
        (source / 'attempts/abc-01/server.log').write_text('preserved log')
        (source / 'attempts/abc-01/link').symlink_to('/absent-node-modules')
        h.sealed_json(source / 'job-done.json', {'status': 'passed', 'model': 'iq3-control',
                      'task': 'javascript/done', 'attempt': 'attempts/abc-01'})
        h.sealed_json(source / 'job-failed.json', {'status': 'failed', 'reason': 'vram_reserve'})
        h.atomic_json(source / 'state.json', {'status': 'stopped', 'owned': [],
                      'completed': {'done': 'job-done.json', 'failed': 'job-failed.json'}})
        h.atomic_json(target / 'state.json', {'status': 'prepared', 'owned': [], 'completed': {}})
        return source, target

    def test_only_valid_results_are_reused_with_verified_logs_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp))
            audit = c.reuse_completed(source, target)
            self.assertEqual([r['job'] for r in audit['reused']], ['done'])
            self.assertEqual([r['job'] for r in audit['requeued']], ['failed'])
            self.assertEqual(c.reuse_completed(source, target), audit)
            self.assertEqual(c.tree_fingerprint(source / 'attempts/abc-01'),
                             c.tree_fingerprint(target / 'attempts/abc-01'))
            self.assertTrue((target / 'attempts/abc-01/link').is_symlink())
            self.assertFalse((target / 'job-failed.json').exists())
            self.assertTrue((source / 'job-failed.json').exists())
            self.assertEqual(json.loads((target / 'state.json').read_text())['completed'], {'done': 'job-done.json'})

    def test_changed_safety_or_grader_prevents_import(self):
        for key in ('safety', 'grader'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                source, target = self.fixture(Path(temp))
                manifest = h.read_sealed(target / 'manifest.json')
                if key == 'grader':
                    manifest['snapshot_pins']['grader'] = 'changed'
                else:
                    manifest['safety'] = 'changed'
                h.sealed_json(target / 'manifest.json', manifest)
                with self.assertRaises(ValueError):
                    c.reuse_completed(source, target)

    def test_watchdog_policy_change_requires_explicit_bounded_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp))
            manifest = h.read_sealed(target / 'manifest.json')
            manifest['continuation']['watchdog_recovery'] = copy.deepcopy(c.WATCHDOG_RECOVERY)
            h.sealed_json(target / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'selection policy'):
                c.reuse_completed(source, target)
            audit = c.reuse_completed(source, target, allow_watchdog_recovery=True)
            self.assertTrue(audit['watchdog_recovery_changed'])
            manifest['continuation']['watchdog_recovery']['max_retries'] = 100
            h.sealed_json(target / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'bounded watchdog'):
                c.reuse_completed(source, target, allow_watchdog_recovery=True)

    def test_grader_migration_requires_opt_in_and_revalidation_for_each_import(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp))
            manifest = h.read_sealed(target / 'manifest.json')
            manifest['snapshot_pins']['scripts/instruments/agentic_probe.py'] = 'fixed'
            manifest['snapshot_pins']['agentic-assets/selftest.json'] = 'new-checks'
            h.sealed_json(target / 'manifest.json', manifest)
            with self.assertRaises(ValueError):
                c.reuse_completed(source, target)
            with patch.object(c, 'revalidate_measurement', return_value={'eligible': False}) as regrade:
                audit = c.reuse_completed(source, target, revalidate_grader=True)
            self.assertEqual(regrade.call_count, 1)
            self.assertEqual(audit['reused'], [])
            self.assertEqual(audit['requeued'][0]['reason'], 'grader_revalidation_required_rerun')
            self.assertFalse((target / 'job-done.json').exists())
            (target / 'revalidation').mkdir()
            h.sealed_json(target / 'revalidation/done.json', {'eligible': True})
            with patch.object(c, 'revalidate_measurement', return_value={'eligible': True}):
                audit = c.reuse_completed(source, target, revalidate_grader=True)
            self.assertTrue(audit['grader_revalidated'])
            self.assertEqual([r['job'] for r in audit['reused']], ['done'])

    def test_grader_migration_still_rejects_changes_to_task_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp))
            manifest = h.read_sealed(target / 'manifest.json')
            manifest['snapshot_pins']['agentic-assets/sources/rust/decimal/tests/decimal.rs'] = 'changed'
            h.sealed_json(target / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'grader assets'):
                c.reuse_completed(source, target, revalidate_grader=True)

    def test_revalidation_requeues_false_repairs_or_changed_counts_without_regrading(self):
        from scripts.instruments import agentic_probe as probe
        for change in ({'grading_rounds': 2}, {'expected_passed_tests': 45}, {'timed_out': True}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                source, target = self.fixture(Path(temp))
                assets = target / 'snapshot/agentic-assets'
                assets.mkdir(parents=True)
                h.atomic_json(assets / 'selftest.json', {'test_scope': probe.TEST_SCOPE, 'harness_verified': True,
                              'tasks': [{'task_id': 'javascript/done', 'expected_passed_tests': 44}]})
                row = {'probe': {'complete': True, 'task_success': True, 'grading_rounds': 1,
                                'expected_passed_tests': 44, **change}}
                with patch.object(probe, 'test_suite') as grade:
                    report = c.revalidate_measurement(source, target,
                        {'id': 'done', 'task': 'javascript/done'}, row, 'tree')
                    self.assertFalse(report['eligible'])
                    grade.assert_not_called()
                    self.assertEqual(c.revalidate_measurement(source, target,
                        {'id': 'done', 'task': 'javascript/done'}, row, 'tree'), report)
                    with self.assertRaisesRegex(ValueError, 'inputs changed'):
                        c.revalidate_measurement(source, target,
                            {'id': 'done', 'task': 'javascript/done'}, row, 'different-tree')

    def test_queue_reorder_requires_opt_in_and_preserves_every_job_definition(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp))
            manifest = h.read_sealed(target / 'manifest.json')
            manifest['queue'].reverse()
            manifest['continuation']['queue_order'] = 'model-first'
            h.sealed_json(target / 'manifest.json', manifest)
            with self.assertRaises(ValueError):
                c.reuse_completed(source, target)
            audit = c.reuse_completed(source, target, allow_queue_reorder=True)
            self.assertTrue(audit['queue_order_changed'])
            self.assertEqual(len(audit['reused']), 1)
        for change in ('duplicate', 'missing', 'config'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                source, target = self.fixture(Path(temp))
                manifest = h.read_sealed(target / 'manifest.json')
                if change == 'duplicate':
                    manifest['queue'].append(manifest['queue'][0])
                elif change == 'missing':
                    manifest['queue'].pop()
                else:
                    manifest['queue'][0]['configs'] = [{'context': 32768}]
                h.sealed_json(target / 'manifest.json', manifest)
                with self.assertRaises(ValueError):
                    c.reuse_completed(source, target, allow_queue_reorder=True)

    def test_live_source_or_changed_import_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp))
            c.reuse_completed(source, target)
            (target / 'attempts/abc-01/server.log').write_text('changed')
            with self.assertRaises(ValueError):
                c.reuse_completed(source, target)
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp))
            state = json.loads((source / 'state.json').read_text())
            state['status'] = 'running'
            h.atomic_json(source / 'state.json', state)
            with self.assertRaises(ValueError):
                c.reuse_completed(source, target)


class BoundaryHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source, self.target = Path(self.temp.name) / 'source', Path(self.temp.name) / 'target'
        self.source.mkdir(); self.target.mkdir()
        h.sealed_json(self.source / 'manifest.json', {'queue': [{'id': 'current'}]})
        self.source_state = {'status': 'running', 'completed': {}, 'owned': [{'pid': 123}],
                             'coordinator': {'pid': 123, 'start': 'identity'}}
        h.atomic_json(self.source / 'state.json', self.source_state)
        self.deadline = time.time() + 1000

    def finish(self):
        h.sealed_json(self.source / 'job-current.json', {'status': 'passed', 'probe': {'task_success': False}})
        self.source_state['completed']['current'] = 'job-current.json'
        h.atomic_json(self.source / 'state.json', self.source_state)

    def import_results(self, source, target, *, allow_queue_reorder):
        self.assertTrue(allow_queue_reorder)
        self.assertEqual((source, target), (self.source, self.target))
        h.sealed_json(target / 'RECOVERY.json', {'source': str(source), 'reused': ['current']})

    def test_waits_for_record_then_stops_and_imports_only_after_coordinator_exits(self):
        ticks = [0]
        def step(_):
            ticks[0] += 1
            if ticks[0] == 1:
                self.assertFalse((self.source / 'STOP').exists())
                self.finish()
            elif ticks[0] == 2:
                self.assertTrue((self.source / 'STOP').exists())
                self.source_state.update(status='stopped', reason='requested_stop', owned=[])
                h.atomic_json(self.source / 'state.json', self.source_state)
        def identity(_):
            return None if ticks[0] >= 3 else 'identity'
        with patch.object(c.time, 'sleep', side_effect=step), patch.object(h, 'proc_identity', side_effect=identity), \
                patch.object(c, 'reuse_completed', side_effect=self.import_results) as reuse:
            self.assertTrue(c.handoff_at_boundary(self.source, self.target, 'current', self.deadline))
            self.assertEqual(reuse.call_count, 1)
            self.assertTrue(c.handoff_at_boundary(self.source, self.target, 'current', self.deadline))
            self.assertEqual(reuse.call_count, 1)
        self.assertGreaterEqual(ticks[0], 3)

    def test_source_safety_fault_blocks_even_when_boundary_was_completed(self):
        for reason in ('gpu_fault', 'memory_pressure_unresolved', 'budget_expired', 'requested_stop'):
            with self.subTest(reason=reason):
                self.finish()
                self.source_state.update(status='stopped', reason=reason, owned=[])
                h.atomic_json(self.source / 'state.json', self.source_state)
                with patch.object(c, 'reuse_completed') as reuse:
                    self.assertFalse(c.handoff_at_boundary(self.source, self.target, 'current', self.deadline))
                    reuse.assert_not_called()
                self.assertFalse((self.source / 'STOP').exists())

    def test_external_stop_and_destination_stop_do_not_stop_or_restart_source(self):
        (self.source / 'STOP').touch()
        self.assertFalse(c.handoff_at_boundary(self.source, self.target, 'current', self.deadline))
        self.assertFalse((self.target / 'HANDOFF-STOP.json').exists())
        (self.target / 'STOP').touch()
        self.assertFalse(c.handoff_at_boundary(self.source, self.target, 'current', self.deadline))

    def test_deadline_and_incomplete_record_do_not_interrupt_active_task(self):
        h.atomic_json(self.source / 'job-current.json', {'status': 'passed'})
        clock = [self.deadline - .1]
        with patch.object(c.time, 'time', side_effect=lambda: clock[0]), \
                patch.object(c.time, 'sleep', side_effect=lambda _: clock.__setitem__(0, clock[0] + 1)):
            self.assertFalse(c.handoff_at_boundary(self.source, self.target, 'current', self.deadline))
        self.assertFalse((self.source / 'STOP').exists())

    def test_shutdown_timeout_is_fixed_across_restarts(self):
        self.finish()
        h.sealed_json(self.target / 'HANDOFF-STOP.json', {'requested_at': time.time() - 181})
        with patch.object(h, 'proc_identity', return_value='identity'), patch.object(c, 'reuse_completed') as reuse:
            self.assertFalse(c.handoff_at_boundary(self.source, self.target, 'current', self.deadline))
            reuse.assert_not_called()
        self.assertEqual(json.loads((self.target / 'stage-status.json').read_text())['reason'], 'source_shutdown_timeout')


if __name__ == '__main__':
    unittest.main()
