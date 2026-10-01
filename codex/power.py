#!/usr/bin/env python3
"""Exact unconditional power of a two-sided Fisher test, equal arm sizes."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, hypergeom

from runaways import ci


def rejection_matrix(n, alpha):
    reject = np.zeros((n + 1, n + 1), dtype=bool)
    for total in range(2 * n + 1):
        x = np.arange(max(0, total - n), min(n, total) + 1)
        mass = hypergeom.pmf(x, 2 * n, total, n)
        p = (mass[None, :] * (mass[None, :] <= mass[:, None] * (1 + 1e-12))).sum(axis=1)
        reject[x, total - x] = p <= alpha
    return reject


def power(n, p0, p1, reject):
    x = np.arange(n + 1)
    return min(1.0, float((np.outer(binom.pmf(x, n, p0), binom.pmf(x, n, p1)) * reject).sum()))


def main():
    alpha = .05 / 6
    scenarios = [(0.01, 0.30), (0.01, 0.20), (0.01, 0.10), (0.01, 0.05), (0.02, 0.10)]
    results = []
    for n in [32, 64, 128, 256, 512]:
        reject = rejection_matrix(n, alpha)
        for p0, p1 in scenarios:
            results.append(dict(n_per_arm_D=n, null_rate_G=p0, alternate_rate_G=p1,
                                alpha_D=alpha, exact_power_D=power(n, p0, p1, reject),
                                zero_event_upper_95_D=ci(0, n)[1]))
    Path('codex/power.json').write_text(json.dumps(results, indent=2) + '\n')
    for r in results:
        print('D:', r)


if __name__ == '__main__':
    main()
