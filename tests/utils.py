import numpy as np
from pytest import approx


def expect_identical(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a).union(b):
            expect_identical(a[key], b[key])
    else:
        assert np.all(a == b)


def expect_equal(current, target, tolerance=None):
    if isinstance(current, dict) and isinstance(target, dict):
        return expect_equal(
            np.array(list(current.values())), np.array(list(target.values())), tolerance
        )

    assert current == approx(target, rel=tolerance, abs=tolerance)


def expect_false(value):
    assert not value
