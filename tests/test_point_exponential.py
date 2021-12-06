import numpy as np
from utils import expect_equal, expect_false, expect_identical

from ebnmpy.ebnm import ebnm
from ebnmpy.ebnm_fns import ebnm_point_exponential
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

pe_res = ebnm(x, s, prior_family="point_exponential")


def test_basic_functionality_works():
    pe_res2 = ebnm_point_exponential(x, s)
    expect_identical(pe_res, pe_res2)
    expect_equal(pe_res[g_ret_str()], true_g, tolerance=0.1)


def test_mode_estimation_works():
    pe_res2 = ebnm_point_exponential(x, s, mode="estimate")
    expect_equal(pe_res2[g_ret_str()], true_g, tolerance=0.5)
    expect_false(pe_res2[g_ret_str()]["shift"][0] == true_mode)


def test_fixing_the_scale_works():
    pe_res2 = ebnm_point_exponential(x, s, scale=true_scale)
    expect_equal(pe_res2[g_ret_str()], true_g, tolerance=0.1)
    expect_equal(pe_res2[g_ret_str()]["scale"][1], true_scale)


def test_fixing_g_works():
    pe_res2 = ebnm_point_exponential(x, s, g_init=pe_res[g_ret_str()], fix_g=True)
    expect_identical(pe_res[g_ret_str()], pe_res2[g_ret_str()])
    expect_equal(pe_res[llik_ret_str()], pe_res2[llik_ret_str()])


def test_initializing_g_works():
    pe_res2 = ebnm_point_exponential(x, s, g_init=true_g)

    print()
    print(f"{pe_res2[llik_ret_str()]=}")
    expect_equal(pe_res[llik_ret_str()], pe_res2[llik_ret_str()])


def test_output_parameter_works():
    pe_res = ebnm_point_exponential(x, s, output=("fitted_g",))
    expect_identical(tuple(pe_res), ("fitted_g",))


def test_can_fix_g_with_one_component():
    g_init1 = gammamix(pi=1, shape=1, scale=true_scale, shift=true_mode)
    pe_res1 = ebnm_point_exponential(x, s, g_init=g_init1, fix_g=True)

    g_init2 = gammamix(pi=(0, 1), shape=(1, 1), scale=(0, true_scale), shift=(true_mode, true_mode))
    pe_res2 = ebnm_point_exponential(x, s, g_init=g_init2, fix_g=True)

    expect_equal(pe_res1["log_likelihood"], pe_res2["log_likelihood"])


def test_null_case_estimates_pi0_equals_1():
    np.random.seed(0)
    x_ = rnorm(n)
    pe_res_ = ebnm_point_exponential(x_, s=1, scale=1)
    expect_equal(pe_res_[g_ret_str()]["pi"][0], 1)
