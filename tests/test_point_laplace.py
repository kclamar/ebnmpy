import numpy as np
from utils import expect_equal, expect_false, expect_identical

from ebnmpy.ebnm import ebnm
from ebnmpy.ebnm_fns import ebnm_point_laplace
from ebnmpy.output import g_ret_str, samp_arg_str, samp_ret_str
from ebnmpy.r_utils.stats import rnorm

n = 1000
np.random.seed(0)
s = rnorm(n, 1, 0.1)
x = np.concatenate((rnorm(n // 2, 0, 10 + 0.1), rnorm(n // 2, 0, 0.1)))

true_pi0 = 0.5
true_mean = 0
true_sd = 10
true_g = dict(pi=(true_pi0, 1 - true_pi0), mean=(true_mean,) * 2, sd=(0, true_sd))


def test_basic_functionality():
    pn_res = ebnm(x, s, prior_family="point_normal")
    pn_res2 = ebnm_point_laplace(x, s)
    expect_identical(pn_res, pn_res2)
    expect_equal(pn_res[g_ret_str()], true_g, tolerance=0.1)


def test_mode_estimation():
    pn_res = ebnm_point_laplace(x, s, mode="estimate")
    expect_equal(pn_res[g_ret_str()], true_g, tolerance=0.1)
    expect_false(pn_res[g_ret_str()]["mean"][0] == true_mean)


def test_fix_sd():
    pn_res = ebnm_point_laplace(x, s, scale=true_sd)
    expect_equal(pn_res[g_ret_str()], true_g, tolerance=0.1)
    expect_equal(pn_res[g_ret_str()]["sd"][1], true_sd, tolerance=1e-8)


def test_fix_g():
    pn_res = ebnm_point_laplace(x, s, g_init=true_g, fix_g=True)
    expect_equal(pn_res[g_ret_str()], true_g)


def test_output_parameter():
    # pass
    pn_res = ebnm_point_laplace(x, s, output=(samp_arg_str(),))
    expect_identical(tuple(pn_res), (samp_ret_str(),))


def test_compute_summary_results():
    # TODO
    pass
