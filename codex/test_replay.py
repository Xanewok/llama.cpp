import unittest
import os
from fractions import Fraction as F

from replay import Lookup, alpha_for_tau, expected_cost, gate


def verified_law(p, q, proposal_law=None):
    proposal_law = proposal_law or q
    mass = [max(a-b, F(0)) for a, b in zip(p, q)]
    residual = [v / sum(mass) for v in mass] if sum(mass) else p
    out = [F(0)] * len(p)
    for token, chance in enumerate(proposal_law):
        if not chance:
            continue
        accept = min(F(1), p[token] / q[token])
        out[token] += chance * accept
        for i in range(len(p)):
            out[i] += chance * (1-accept) * residual[i]
    return out


class Tests(unittest.TestCase):
    def test_prefix_only_and_recent(self):
        lookup = Lookup()
        for token in [1, 2, 3, 4, 5, 9, 1, 2, 3]:
            lookup.append(token)
        self.assertEqual(lookup.propose(), (3, [4, 5, 9, 1, 2, 3], 6))
        before = lookup.propose()
        future = [99] * 100
        self.assertEqual(lookup.propose(), before)
        self.assertFalse(gate(before[0], before[2], 3, 8))
        for token in future:
            lookup.append(token)
        self.assertEqual(lookup.propose()[0], 32)
        self.assertEqual(lookup.propose()[2], 1)

    def test_self_suffix_is_excluded(self):
        lookup = Lookup()
        for token in [1, 2, 3, 4]:
            lookup.append(token)
        self.assertEqual(lookup.propose(), (0, [], 0))

    def test_disabled_gate_and_wrong_round(self):
        row = {'labs': 'r'*100, 'match': [32]*100, 'run': [0]*100, 'available': [16]*100}
        alpha = alpha_for_tau(4., 5)
        base, hybrid, count, wrong = expected_cost(row, 5, alpha, 99, 16, 1.5)
        self.assertEqual(base, hybrid)
        self.assertEqual(count, 0)
        base, hybrid, count, wrong = expected_cost(row, 5, alpha, 32, 16, 1.5)
        self.assertEqual(hybrid, 150.)
        self.assertEqual(wrong, 100.)
        self.assertGreater(hybrid, base)

    def test_exact_law_with_history_selected_proposal(self):
        for p in ([F(3, 10), F(7, 10)], [F(0), F(1)], [F(1), F(0)]):
            for q in ([F(1), F(0)], [F(0), F(1)], [F(1, 2), F(1, 2)]):
                proposal = [F(1), F(0)] if os.getenv('HQ_TEST_STALE_Q') and q == [F(1, 2), F(1, 2)] else None
                self.assertEqual(verified_law(p, q, proposal), p)
        # For each accepted history, q may change; the identity holds conditionally.
        heads = [F(3, 10), F(7, 10)]
        conditional = [[F(1, 4), F(3, 4)], [F(4, 5), F(1, 5)]]
        joint = [heads[i] * verified_law(conditional[i], [F(1-i), F(i)])[j]
                 for i in range(2) for j in range(2)]
        self.assertEqual(joint, [heads[i]*conditional[i][j] for i in range(2) for j in range(2)])

    def test_negative_control_stale_q(self):
        p, q = [F(3, 10), F(7, 10)], [F(1, 2), F(1, 2)]
        self.assertNotEqual(verified_law(p, q, [F(1), F(0)]), p)
        # Switching to delta on sampled-token disagreement, retaining the old q otherwise,
        # produces 0.45 rather than p(0)=0.30.
        delta = verified_law(p, [F(1), F(0)])
        retained = verified_law(p, q, [F(1), F(0)])
        selected = [(a+b)/2 for a, b in zip(delta, retained)]
        self.assertEqual(selected[0], F(9, 20))
        self.assertNotEqual(selected, p)


if __name__ == '__main__':
    unittest.main()
