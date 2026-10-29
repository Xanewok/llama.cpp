import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import ROOT
from scripts.instruments import agentic_campaign as agentic
from scripts.instruments import chat_templates as templates
from scripts.instruments import hillclimb as h


class TemplateOverrideTests(unittest.TestCase):
    def test_only_new_bartowski_jobs_select_the_adapter(self):
        jobs = h.validate_profiles(agentic.task_queue())
        for job in jobs:
            config = job['configs'][0]
            if job['model'] == 'bartowski-iq4-xs':
                self.assertEqual(config['chat_template'], templates.BARTOWSKI_PREFIX)
            else:
                self.assertNotIn('chat_template', config)
        bart = next(job for job in jobs if job['model'] == 'bartowski-iq4-xs')
        for key, value in [('model', 'iq3-control'), ('configs', [{'chat_template': '/tmp/other.jinja'}])]:
            with self.assertRaisesRegex(ValueError, 'pinned leading-instructions adapter'):
                h.validate_profiles([{**bart, key: value}])

    def test_changed_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'pinned source'):
            templates.bartowski_template('changed')

    def test_launch_uses_per_attempt_override_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'artifacts').mkdir()
            out = root / 'attempt'
            out.mkdir()
            (root / 'artifacts/bartowski-iq4-xs.json').write_text(json.dumps({'chat_template': 'original'}))
            runner = h.Runner.__new__(h.Runner)
            runner.path = root
            with patch.object(templates, 'bartowski_template', return_value='patched') as adapt:
                args, info = runner.template_override({'id': 'bartowski-iq4-xs'},
                    {'chat_template': templates.BARTOWSKI_PREFIX}, out)
            adapt.assert_called_once_with('original')
            self.assertEqual(args, ['--jinja', '--chat-template-file', str(out / 'chat-template.jinja')])
            self.assertEqual(info['effective_sha256'], hashlib.sha256(b'patched').hexdigest())
            self.assertEqual(info['source_sha256'], templates.BARTOWSKI_TEMPLATE_SHA256)
            self.assertEqual(runner.template_override({'id': 'iq3-control'}, {}, out), ([], None))
            with self.assertRaisesRegex(h.RejectArm, 'unsupported_chat_template'):
                runner.template_override({'id': 'iq3-control'}, {'chat_template': templates.BARTOWSKI_PREFIX}, out)


@unittest.skipUnless(os.environ.get('QWEN38_TEMPLATE_TESTS') == '1', 'requires pinned GGUF and llama.cpp renderer')
class RealTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(h.LLAMA / 'gguf-py'))
        from gguf import GGUFReader
        model = h.ROOT / '.cache/hillclimb-models/bartowski-iq4-xs/Qwen3.8-27B-IQ4_XS.gguf'
        cls.source = GGUFReader(model).fields['tokenizer.chat_template'].contents()
        cls.fixed = templates.bartowski_template(cls.source)

    def render(self, template, messages, *, common=False, failure=False, **settings):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, request, output = root / 'template.jinja', root / 'request.json', root / 'output.txt'
            path.write_text(template)
            payload = {'messages': messages, 'tools': [], 'add_generation_prompt': True,
                       'reasoning_effort': 'xhigh', **settings}
            request.write_text(json.dumps(payload))
            args = [str(h.LLAMA / 'build/bin/test-chat-template'), '--json', str(request),
                    '--output', str(output), str(path)]
            if not common:
                args.append('--no-common')
            result = subprocess.run(args, capture_output=True, text=True, timeout=30,
                                    preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
            if failure:
                self.assertNotEqual(result.returncode, 0)
                return result.stderr + result.stdout
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return output.read_text()

    def test_only_the_instruction_normalization_is_added(self):
        self.assertEqual(self.fixed.replace(templates.LEADING_INSTRUCTIONS, '', 1), self.source)

    def test_existing_single_instruction_and_user_only_prompts_are_identical(self):
        for prefix in [[], [{'role': 'system', 'content': ' instructions '}]]:
            messages = prefix + [{'role': 'user', 'content': 'question'}]
            for common in [False, True]:
                self.assertEqual(self.render(self.source, messages, common=common),
                                 self.render(self.fixed, messages, common=common))

    def test_multiple_system_and_developer_messages_merge_in_order(self):
        messages = [{'role': 'system', 'content': ' FIRST '},
                    {'role': 'developer', 'content': [{'type': 'text', 'text': 'SECOND'}]},
                    {'role': 'system', 'content': 'THIRD'},
                    {'role': 'user', 'content': 'question'}]
        merged = [{'role': 'system', 'content': 'FIRST\nSECOND\nTHIRD'}, messages[-1]]
        for common in [False, True]:
            self.assertEqual(self.render(self.fixed, messages, common=common),
                             self.render(self.source, merged, common=common))
        error = self.render(self.source, messages, failure=True)
        self.assertTrue('Unexpected message role' in error or 'System message must be at the beginning' in error)

    def test_tools_reasoning_and_thinking_mode_are_preserved(self):
        tool = {'type': 'function', 'function': {'name': 'run', 'parameters': {'type': 'object',
                'properties': {'cmd': {'type': 'string'}}}}}
        messages = [{'role': 'system', 'content': 'FIRST'}, {'role': 'developer', 'content': 'SECOND'},
                    {'role': 'user', 'content': 'question'},
                    {'role': 'assistant', 'content': '', 'reasoning_content': 'THOUGHT',
                     'tool_calls': [{'type': 'function', 'id': 'c1', 'function': {
                         'name': 'run', 'arguments': {'cmd': 'echo hello'}}}]},
                    {'role': 'tool', 'tool_call_id': 'c1', 'content': 'hello'}]
        merged = [{'role': 'system', 'content': 'FIRST\nSECOND'}, *messages[2:]]
        for settings in [{}, {'enable_thinking': False}, {'preserve_thinking': False}, {'reasoning_effort': 'medium'}]:
            self.assertEqual(self.render(self.fixed, messages, tools=[tool], **settings),
                             self.render(self.source, merged, tools=[tool], **settings))

    def test_late_instructions_are_not_hoisted_or_silently_discarded(self):
        for role in ['system', 'developer']:
            messages = [{'role': 'system', 'content': 'FIRST'}, {'role': 'user', 'content': 'question'},
                        {'role': role, 'content': 'LATE'}]
            self.render(self.fixed, messages, failure=True)

    def test_developer_only_and_empty_leading_blocks(self):
        messages = [{'role': 'developer', 'content': ''}, {'role': 'system', 'content': None},
                    {'role': 'developer', 'content': 'ONLY'}, {'role': 'user', 'content': 'question'}]
        self.assertEqual(self.render(self.fixed, messages), self.render(self.source,
            [{'role': 'system', 'content': 'ONLY'}, messages[-1]]))
        invalid = copy.deepcopy(messages)
        invalid[0]['content'] = [{'type': 'image', 'image': 'bad'}]
        self.assertIn('cannot contain images', self.render(self.fixed, invalid, failure=True))


if __name__ == '__main__':
    unittest.main()


class SwiftDropTracesTests(unittest.TestCase):
    SOURCE = (ROOT / 'results/2026-09-12-mixed-pair/campaign-swift-fullsuite-20260915'
              / 'swift-iq4-xs/attempts/66ebeeaec6a7043b-01/chat-template.jinja')

    def test_changed_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'pinned source'):
            templates.swift_drop_traces_template('changed')

    def test_pinned_source_still_on_disk(self):
        if not self.SOURCE.exists():
            self.skipTest('pinned Swift template not present')
        digest = hashlib.sha256(self.SOURCE.read_text().encode()).hexdigest()
        self.assertEqual(digest, templates.SWIFT_TEMPLATE_SHA256)

    def test_only_the_guard_line_changes(self):
        if not self.SOURCE.exists():
            self.skipTest('pinned Swift template not present')
        source = self.SOURCE.read_text()
        patched = templates.swift_drop_traces_template(source)
        self.assertNotIn(templates.PRESERVE_THINKING_GUARD, patched)
        self.assertEqual(patched.count(templates.DROP_THINKING_GUARD), 1)
        differing = [(a, b) for a, b in zip(source.splitlines(), patched.splitlines()) if a != b]
        self.assertEqual(len(differing), 1)
        self.assertEqual(len(source.splitlines()), len(patched.splitlines()))

    def test_in_flight_chain_still_keeps_its_traces(self):
        # The `loop.index0 > ns.last_query_index` arm is what protects the current
        # tool-call chain; losing it would strip reasoning the model is still acting on.
        self.assertIn('loop.index0 > ns.last_query_index', templates.DROP_THINKING_GUARD)
