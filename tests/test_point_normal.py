import numpy as np
from utils import expect_equal, expect_false, expect_identical

from ebnmpy.ashr import normalmix
from ebnmpy.ebnm import ebnm
from ebnmpy.ebnm_fns import ebnm_point_normal
from ebnmpy.estimators import PointNormalEBNM
from ebnmpy.output import g_ret_str, llik_ret_str, samp_arg_str, samp_ret_str
from ebnmpy.r_utils.stats import rnorm

n = 1000
np.random.seed(0)
s = rnorm(n, 1, 0.1)
x = np.concatenate((rnorm(n // 2, 0, 10), np.zeros(n // 2))) + rnorm(n, 0, s)

true_pi0 = 0.5
true_mean = 0
true_sd = 10
true_g = dict(pi=(true_pi0, 1 - true_pi0), mean=(true_mean,) * 2, sd=(0, true_sd))


def test_basic_functionality():
    pn_res = ebnm(x, s, prior_family="point_normal")
    pn_res2 = ebnm_point_normal(x, s)
    est = PointNormalEBNM().fit(x=x, s=s)

    for key, val in pn_res.items():
        expect_identical(val, getattr(est, key + "_"))

    expect_identical(pn_res, pn_res2)
    expect_equal(pn_res[g_ret_str()], true_g, tolerance=0.1)


def test_mode_estimation():
    pn_res = ebnm_point_normal(x, s, mode="estimate")
    expect_equal(pn_res[g_ret_str()], true_g, tolerance=0.1)
    expect_false(pn_res[g_ret_str()]["mean"][0] == true_mean)


def test_fix_sd():
    pn_res = ebnm_point_normal(x, s, scale=true_sd)
    expect_equal(pn_res[g_ret_str()], true_g, tolerance=0.1)
    expect_equal(pn_res[g_ret_str()]["sd"][1], true_sd, tolerance=1e-8)


def test_fix_g():
    pn_res = ebnm_point_normal(x, s, g_init=true_g, fix_g=True)
    expect_equal(pn_res[g_ret_str()], true_g)


def test_output_parameter():
    pn_res = ebnm_point_normal(x, s, output=(samp_arg_str(),))
    expect_identical(tuple(pn_res), (samp_ret_str(),))


def test_null_case_estimates_pi0_equals_1():
    x_ = rnorm(n, sd=0.5)
    pn_res = ebnm_point_normal(x_, s=1)
    expect_equal(pn_res[g_ret_str()]["pi"][0], 1)


def test_very_large_observations_give_reasonable_results():
    scl = 1e8
    pn_res = ebnm_point_normal(x, s, mode="estimate")
    pn_res_lg = ebnm_point_normal(scl * x, scl * s, mode="estimate")

    expect_equal(pn_res[g_ret_str()]["pi"][0], pn_res_lg[g_ret_str()]["pi"][0])
    expect_equal(scl * pn_res[g_ret_str()]["sd"][1], pn_res_lg[g_ret_str()]["sd"][1])
    expect_equal(scl * pn_res[g_ret_str()]["mean"][0], pn_res_lg[g_ret_str()]["mean"][0])


def test_very_small_observations_give_reasonable_results():
    scl = 1e-8
    pn_res = ebnm_point_normal(x, s, mode="estimate")
    pn_res_lg = ebnm_point_normal(scl * x, scl * s, mode="estimate")

    expect_equal(pn_res[g_ret_str()]["pi"][0], pn_res_lg[g_ret_str()]["pi"][0])
    expect_equal(scl * pn_res[g_ret_str()]["sd"][1], pn_res_lg[g_ret_str()]["sd"][1])
    expect_equal(scl * pn_res[g_ret_str()]["mean"][0], pn_res_lg[g_ret_str()]["mean"][0])


def test_g_init_with_pi0_equals_0_or_pi0_equals_1_isnt_a_dealbreaker():
    pn_res = ebnm_point_normal(x, s)

    bad_g = normalmix((1, 0), (0, 0), (0, true_sd))
    pn_res2 = ebnm_point_normal(x, s, g_init=bad_g, fix_g=False)
    expect_equal(pn_res[llik_ret_str()], pn_res2[llik_ret_str()])

    bad_g = normalmix((0, 1), (0, 0), (0, true_sd))
    pn_res3 = ebnm_point_normal(x, s, g_init=bad_g, fix_g=False)
    expect_equal(pn_res[llik_ret_str()], pn_res3[llik_ret_str()])
