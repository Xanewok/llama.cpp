#!/usr/bin/env python3
"""Exact algebra oracle for the proposed sampling contract, not an engine test."""
from fractions import Fraction as F
from itertools import product


def distributions(total):
    return [tuple(F(x, total) for x in row)
            for row in product(range(total + 1), repeat=3) if sum(row) == total]


def verified_law(p, actual_q, recorded_q):
    acceptance = [min(F(1), px / qx) if qx else F(0)
                  for px, qx in zip(p, recorded_q)]
    accepted = [qx * ax for qx, ax in zip(actual_q, acceptance)]
    rejected = 1 - sum(accepted)
    residual = [max(px - qx, F(0)) for px, qx in zip(p, recorded_q)]
    mass = sum(residual)
    if not mass:
        return tuple(a + rejected * px for a, px in zip(accepted, p))
    return tuple(a + rejected * r / mass for a, r in zip(accepted, residual))


def require_law(p, actual_q, recorded_q):
    assert sum(p) == sum(actual_q) == sum(recorded_q) == 1
    assert verified_law(p, actual_q, recorded_q) == p, "proposal/recorded-q law mismatch"


def must_fail(check):
    try:
        check()
    except AssertionError:
        return
    raise AssertionError("negative control escaped")


def main():
    ds = distributions(5)
    for p, q in product(ds, repeat=2):
        require_law(p, q, q)
    p = (F(1, 2), F(1, 3), F(1, 6))
    q = (F(1, 4), F(3, 4), F(0))
    # Correct RNG values do not repair a different proposal normalization or ID map.
    must_fail(lambda: require_law(p, q, (F(1, 2), F(1, 2), F(0))))
    must_fail(lambda: require_law(p, q, (q[1], q[0], q[2])))
    must_fail(lambda: require_law(p, q, (F(1), F(0), F(0))))
    # Shared uniform: x=0 for u<1/4; x=1 otherwise. Accept x=1 when u<4/9.
    # Its accepted x=1 mass is 4/9-1/4, not min(p[1],q[1])=1/3.
    must_fail(lambda: assert_equal(F(4, 9) - F(1, 4), min(p[1], q[1])))
    print(f"M: {len(ds) ** 2} exact rational p/q cases passed; 4 negative controls rejected")
    print("D: accepted mass min(p,q) + rejection residual max(p-q,0) equals p")
    print("No claim about floating-point CUDA sampling, model logits, RNG implementation or KV state.")


def assert_equal(a, b):
    assert a == b, "correlated uniforms violate the acceptance mass"


if __name__ == "__main__":
    main()
