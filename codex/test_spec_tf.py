import unittest

import numpy as np

from spec_tf import estimate_rows, harmonic, paired


K = 5
V = 3
N = 180000


def trees(rng, biased=False):
    nodes = sum(V ** i for i in range(K + 1))
    p = rng.dirichlet(np.full(V, 0.7), nodes)
    q = rng.dirichlet(np.full(V, 0.7), nodes)
    if biased:
        p[0] = [0.90, 0.05, 0.05]
        q[0] = [0.05, 0.475, 0.475]
        # The accepted root law overweights the two rare target branches.
        for node in range(1, nodes):
            parent = node
            while parent > V:
                parent = (parent - 1) // V
            p[node] = [0.96, 0.02, 0.02]
            q[node] = [0.02, 0.49, 0.49] if parent == 1 else p[node]
    return p, q


def categorical(rng, probs):
    return (rng.random(len(probs))[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)


def target_paths(rng, p, q):
    node = np.zeros(N, dtype=int)
    ps, qs, xs = [], [], []
    for _ in range(K):
        ps.append(p[node])
        qs.append(q[node])
        x = categorical(rng, p[node])
        xs.append(x)
        node = node * V + 1 + x
    return np.stack(ps, axis=1), np.stack(qs, axis=1), np.stack(xs, axis=1)


def rejection(rng, p, q):
    node = np.zeros(N, dtype=int)
    alive = np.ones(N, dtype=bool)
    lengths = np.ones(N)
    emitted = np.full(N, -1, dtype=int)
    first = np.full(N, -1, dtype=int)
    for step in range(K):
        d = categorical(rng, q[node])
        px, qx = p[node, d], q[node, d]
        accept = alive & ((px >= qx) | (rng.random(N) < px / qx))
        rejected = alive & ~accept
        if rejected.any():
            residual = np.maximum(0, p[node[rejected]] - q[node[rejected]])
            residual /= residual.sum(axis=1, keepdims=True)
            emitted[rejected] = categorical(rng, residual)
        if step == 0:
            first[accept] = d[accept]
            first[rejected] = emitted[rejected]
        lengths += accept
        alive = accept
        node = node * V + 1 + d
    emitted[alive] = categorical(rng, p[node[alive]])
    return lengths, first


class SpecTFTest(unittest.TestCase):
    def check_tree(self, seed, biased):
        rng = np.random.default_rng(seed)
        p, q = trees(rng, biased)
        ps, qs, xs = target_paths(rng, p, q)
        ids = np.broadcast_to(np.arange(V), ps.shape)
        beta, _, estimate = estimate_rows(ids, ps, ids, qs, xs)
        lengths, _ = rejection(rng, p, q)
        se = np.sqrt((estimate.var(ddof=1) + lengths.var(ddof=1)) / N)
        error = abs(estimate.mean() - lengths.mean())
        print(f"seed={seed} biased={biased} estimator={estimate.mean():.8f} MC={lengths.mean():.8f} combined_SE={se:.8f} z={error / se:.3f}")
        self.assertLess(error, 3 * se)
        if biased:
            broken = 1 + np.cumprod(beta, axis=1).sum(axis=1)
            broken_se = np.sqrt((broken.var(ddof=1) + lengths.var(ddof=1)) / N)
            z = abs(broken.mean() - lengths.mean()) / broken_se
            print(f"negative control plain_product={broken.mean():.8f} z={z:.3f}")
            self.assertGreater(z, 3)

    def test_random_rejection_trees(self):
        for seed in (13, 73, 1729):
            self.check_tree(seed, False)

    def test_biased_product_negative_control(self):
        self.check_tree(42, True)

    def test_target_law_of_rejection(self):
        rng = np.random.default_rng(99)
        p, q = trees(rng)
        # The first emitted token, including a residual draw, must have the target law.
        p[:] = [0.6, 0.3, 0.1]
        _, first = rejection(rng, p, q)
        freq = np.bincount(first, minlength=V) / N
        se = np.sqrt(p[0] * (1 - p[0]) / N)
        self.assertTrue(np.all(np.abs(freq - p[0]) < 3 * se))

    def test_identical_and_fixed_weights(self):
        base = np.array([1.1, 2.0, 5.0])
        arm = np.array([1.5, 1.8, 5.4])
        self.assertEqual(paired(base, base), 0.0)
        self.assertAlmostEqual(paired(base, arm), np.average(arm - base, weights=1 / base))
        self.assertAlmostEqual(harmonic(base), np.average(base, weights=1 / base))


if __name__ == "__main__":
    unittest.main()
