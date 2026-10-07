import unittest
from pathlib import Path

from gpu_payload import journal_fault


class JournalTest(unittest.TestCase):
    def test_recorded_fault(self):
        lines = Path('/home/xanewok/repos/qwen38-eval/orchestration/gpuq/fixtures/xid-lines.txt').read_text()
        self.assertEqual(journal_fault(lines, 3656617), 3)

    def test_application_fault_and_bounds(self):
        line = 'NVRM: Xid (PCI:0000:08:00): 31, pid=123, name=round-trace, MMU Fault'
        self.assertEqual(journal_fault(line, 123), 8)
        self.assertEqual(journal_fault(line, 124), 3)
        self.assertEqual(journal_fault('stop_xid_hash_s_G', 123), 0)
        self.assertEqual(journal_fault(line + '\nGPU has fallen off the bus', 123), 3)


if __name__ == '__main__':
    unittest.main()
