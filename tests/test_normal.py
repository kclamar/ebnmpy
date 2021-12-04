import numpy as np
from pytest import approx

from ebnmpy.ebnm import ebnm
from ebnmpy.ebnm_fns import ebnm_normal
from ebnmpy.output import g_ret_str, samp_arg_str, samp_ret_str
from ebnmpy.r_utils.stat import rnorm

n = 1000
np.random.seed(0)
s = rnorm(n, 1, 0.1)
x = rnorm(n, 0, 10 + s)

true_mean = 0
true_sd = 10
true_g = dict(pi=1, mean=true_mean, sd=true_sd)


def expect_identical(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a).union(b):
            expect_identical(a[key], b[key])
    else:
        assert np.all(a == b)


def expect_equal(current, target, tolerance=None):
    if isinstance(current, dict) and isinstance(target, dict):
        return expect_equal(current.values(), target.values(), tolerance)

    assert current == approx(target, rel=tolerance, abs=tolerance)


def expect_false(value):
    assert not value


def test_basic_functionality():
    norm_res = ebnm(x, s, prior_family="normal")
    norm_res2 = ebnm_normal(x, s)
    expect_identical(norm_res, norm_res2)
    expect_equal(norm_res[g_ret_str()], true_g, tolerance=0.2)


def test_mode_estimation():
    norm_res = ebnm_normal(x, s, mode="estimate")
    expect_equal(norm_res[g_ret_str()], true_g, tolerance=0.2)
    expect_false(norm_res[g_ret_str()]["mean"] == true_mean)


def test_fix_sd():
    norm_res = ebnm_normal(x, s, scale=true_sd)
    expect_equal(norm_res[g_ret_str()], true_g, tolerance=0.1)
    expect_equal(norm_res[g_ret_str()]["sd"], true_sd)


def test_fix_g():
    norm_res = ebnm_normal(x, s, g_init=true_g, fix_g=True)
    expect_equal(norm_res[g_ret_str()], true_g)


def test_output_parameter():
    # pass
    norm_res = ebnm_normal(x, s, output=(samp_arg_str(),))
    expect_identical(tuple(norm_res), (samp_ret_str(),))


def test_compute_summary_results():
    # TODO
    pass
