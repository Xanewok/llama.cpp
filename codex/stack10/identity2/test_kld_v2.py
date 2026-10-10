"""CPU checks use recorded GPU evidence and never start the tool."""
import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

import kld_item as old
import kld_item_v2 as fixed
import render_kld_v2


class CanaryFixTests(unittest.TestCase):
    def test_target_engagement_without_fold(self):
        path = fixed.EVAL / 'results/2026-10-09-s10-depth-iq4/canary-a3/0-s10/server.log'
        lines = path.read_text().splitlines()
        target_only = '\n'.join(line for line in lines if 'mtp_fold_decode:' not in line)
        fixed.verify_engagement(target_only)
        for marker in ('fattn_q8_decode:', 'persistent rotations backend=', 'one pinned copy per input'):
            with self.subTest(marker=marker), self.assertRaises(fixed.lib.Fail):
                fixed.verify_engagement('\n'.join(line for line in target_only.splitlines() if marker not in line))

    def test_experiment_preserved(self):
        self.assertEqual(old.ARMS, fixed.ARMS)
        self.assertEqual(old.PLAN, fixed.PLAN)
        with tempfile.TemporaryDirectory(dir=fixed.HERE) as tmp:
            before, after = Path(tmp) / 'old.bin', Path(tmp) / 'new.bin'
            old.prepare(before, [8192, 32768, 90112], 64)
            fixed.prepare(after, [8192, 32768, 90112], 64)
            self.assertEqual(before.read_bytes(), after.read_bytes())
        env_old = old.environment(Path('/frozen'), Path('/input'), Path('/output'), {}, True)
        env_new = fixed.environment(Path('/frozen'), Path('/input'), Path('/output'), {}, True)
        self.assertEqual(env_new.pop('LLAMA_MTP_FOLD_C'), '0')
        env_old.pop('LLAMA_MTP_FOLD_C')
        self.assertEqual(env_old, env_new)
        pairs = {'s9p_to_noise': dict.fromkeys(('mean_kld_D', 'p99_kld_D', 'max_kld_D'), 0.)}
        for name in ('s9p_to_off', 's9p_to_stock', 'off_to_stock'):
            pairs[name] = pairs['s9p_to_noise'].copy()
        for name in pairs:
            changed = {k: v.copy() for k, v in pairs.items()}
            changed[name]['max_kld_D'] = .1
            self.assertEqual(old.verdict(changed), fixed.verdict(changed))
        item = tomllib.loads(render_kld_v2.render())
        self.assertEqual(item['id'], 's10-kld-iq4-v2')
        self.assertEqual(item['priority'], 26)
        self.assertEqual(item['kind'], 'measure')
        self.assertFalse(item['host_sensitive'])
        for stage in ('run', 'canary'):
            self.assertIn(str(fixed.WT / 'work/aa-load/ccx01.sh'), item[stage]['cmd'])
            out = item[stage]['cmd'][item[stage]['cmd'].index('--out') + 1]
            self.assertIn('/s10-kld-iq4-v2/', out)
        pins = set(item['pins']['files'])
        self.assertIn(str(fixed.HERE / 'kld-window-v2.cpp'), pins)
        self.assertIn(str(fixed.HERE / 'kld-v2/CMakeLists.txt'), pins)

    def test_history_pins_unchanged(self):
        baseline = json.loads((fixed.HERE / 'evidence/kld-v2/history-pins-before.json').read_text())
        for name, sha in baseline.items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), sha, name)


if __name__ == '__main__':
    unittest.main()
