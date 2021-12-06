import numpy as np


def expect_identical(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a).union(b):
            expect_identical(a[key], b[key])
    else:
        assert np.all(a == b)


def expect_equal(current, target, tolerance=1.5e-8):
    if isinstance(current, dict) and isinstance(target, dict):
        return expect_equal(
            np.array(list(current.values())), np.array(list(target.values())), tolerance
        )

    mean_abs_diff = np.mean(np.abs(current - target))

    if mean_abs_diff >= tolerance:
        assert mean_abs_diff / np.mean(np.abs(target)) < tolerance


def expect_false(value):
    assert not value
