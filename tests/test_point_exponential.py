import numpy as np
from utils import expect_equal, expect_false, expect_identical

from ebnmpy.ebnm import ebnm
from ebnmpy.ebnm_fns import ebnm_point_exponential
from ebnmpy.estimators import PointExponentialEBNM
from ebnmpy.output import g_ret_str, llik_ret_str
from ebnmpy.point_exponential import gammamix
from ebnmpy.r_utils.stats import rexp, rnorm

n = 1000
np.random.seed(0)
s = rnorm(n, 1, 0.1)
x = np.concatenate((rexp(n // 2, rate=0.1), np.zeros(n // 2))) + rnorm(n, sd=s)

true_pi0 = 0.5
true_scale = 10
true_mode = 0

true_g = gammamix(
    pi=(true_pi0, 1 - true_pi0), shape=(1, 1), scale=(0, true_scale), shift=(true_mode, true_mode)
)

res = ebnm(x, s, prior_family="point_exponential")


def test_basic_functionality_works():
    res2 = ebnm_point_exponential(x, s)
    est = PointExponentialEBNM().fit(x=x, s=s)

    for key, val in res.items():
        expect_identical(val, getattr(est, key + "_"))

    expect_identical(res, res2)
    expect_equal(res[g_ret_str()], true_g, tolerance=0.1)


def test_mode_estimation_works():
    res2 = ebnm_point_exponential(x, s, mode="estimate")
    expect_equal(res2[g_ret_str()], true_g, tolerance=0.5)
    expect_false(res2[g_ret_str()]["shift"][0] == true_mode)


def test_fixing_the_scale_works():
    res2 = ebnm_point_exponential(x, s, scale=true_scale)
    expect_equal(res2[g_ret_str()], true_g, tolerance=0.1)
    expect_equal(res2[g_ret_str()]["scale"][1], true_scale)


def test_fixing_g_works():
    res2 = ebnm_point_exponential(x, s, g_init=res[g_ret_str()], fix_g=True)
    expect_identical(res[g_ret_str()], res2[g_ret_str()])
    expect_equal(res[llik_ret_str()], res2[llik_ret_str()])


def test_initializing_g_works():
    res2 = ebnm_point_exponential(x, s, g_init=true_g)

    print()
    print(f"{res2[llik_ret_str()]=}")
    expect_equal(res[llik_ret_str()], res2[llik_ret_str()])


def test_output_parameter_works():
    res = ebnm_point_exponential(x, s, output=("fitted_g",))
    expect_identical(tuple(res), ("fitted_g",))


def test_can_fix_g_with_one_component():
    g_init1 = gammamix(pi=1, shape=1, scale=true_scale, shift=true_mode)
    res1 = ebnm_point_exponential(x, s, g_init=g_init1, fix_g=True)

    g_init2 = gammamix(pi=(0, 1), shape=(1, 1), scale=(0, true_scale), shift=(true_mode, true_mode))
    res2 = ebnm_point_exponential(x, s, g_init=g_init2, fix_g=True)

    expect_equal(res1["log_likelihood"], res2["log_likelihood"])


def test_null_case_estimates_pi0_equals_1():
    np.random.seed(0)
    x_ = rnorm(n)
    res_ = ebnm_point_exponential(x_, s=1, scale=1)
    expect_equal(res_[g_ret_str()]["pi"][0], 1)
