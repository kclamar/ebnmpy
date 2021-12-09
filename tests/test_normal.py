import numpy as np
from utils import expect_equal, expect_false, expect_identical

from ebnmpy.ebnm import ebnm
from ebnmpy.ebnm_fns import ebnm_normal
from ebnmpy.estimators import NormalEBNM
from ebnmpy.output import g_ret_str, samp_arg_str, samp_ret_str
from ebnmpy.r_utils.stats import rnorm

n = 1000
np.random.seed(0)
s = rnorm(n, 1, 0.1)
x = rnorm(n, 0, 10 + s)

true_mean = 0
true_sd = 10
true_g = dict(pi=1, mean=true_mean, sd=true_sd)


def test_basic_functionality():
    res = ebnm(x, s, prior_family="normal")
    res2 = ebnm_normal(x, s)
    est = NormalEBNM().fit(x=x, s=s)

    for key, val in res.items():
        expect_identical(val, getattr(est, key + "_"))

    expect_identical(res, res2)
    expect_equal(res[g_ret_str()], true_g, tolerance=0.2)


def test_mode_estimation():
    res = ebnm_normal(x, s, mode="estimate")

    expect_equal(res[g_ret_str()], true_g, tolerance=0.2)
    expect_false(res[g_ret_str()]["mean"] == true_mean)


def test_fix_sd():
    res = ebnm_normal(x, s, scale=true_sd)
    expect_equal(res[g_ret_str()], true_g, tolerance=0.1)
    expect_equal(res[g_ret_str()]["sd"], true_sd)


def test_fix_g():
    res = ebnm_normal(x, s, g_init=true_g, fix_g=True)
    expect_equal(res[g_ret_str()], true_g)


def test_output_parameter():
    res = ebnm_normal(x, s, output=(samp_arg_str(),))
    expect_identical(tuple(res), (samp_ret_str(),))
