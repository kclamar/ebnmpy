import numpy as np
from utils import expect_equal, expect_false, expect_identical

from ebnmpy.ebnm import ebnm
from ebnmpy.ebnm_fns import ebnm_point_laplace
from ebnmpy.output import g_ret_str, llik_ret_str
from ebnmpy.r_utils import rep
from ebnmpy.r_utils.stats import rexp, rnorm

n = 1000
np.random.seed(0)
s = rnorm(n, 1, 0.1)
x = np.concatenate((rexp(n // 2, rate=0.1), rep(0, n // 2))) + rnorm(n, sd=s)

true_pi0 = 0.5
true_scale = 10
true_mean = 0

true_g = dict(pi=(true_pi0, 1 - true_pi0), mean=(true_mean,) * 2, scale=(0, true_scale))

np.random.seed(0)
pl_res = ebnm(x, s, prior_family="point_laplace", optmethod="nograd_lbfgsb")


def test_basic_functionality():
    np.random.seed(0)
    pl_res2 = ebnm_point_laplace(x, s)
    expect_identical(pl_res, pl_res2)
    expect_equal(pl_res[g_ret_str()], true_g, tolerance=0.15)


def test_mode_estimation():
    pl_res2 = ebnm_point_laplace(x, s, mode="estimate")
    expect_equal(pl_res2[g_ret_str()], true_g, tolerance=0.5)
    expect_false(pl_res2[g_ret_str()]["mean"][0] == true_mean)


def test_fix_scale():
    pl_res2 = ebnm_point_laplace(x, s, scale=true_scale)
    expect_equal(pl_res2[g_ret_str()], true_g, tolerance=0.1)
    expect_equal(pl_res2[g_ret_str()]["scale"][1], true_scale)


def test_fix_g():
    pl_res2 = ebnm_point_laplace(x, s, g_init=pl_res[g_ret_str()], fix_g=True)
    expect_identical(pl_res[g_ret_str()], pl_res2[g_ret_str()])
    expect_equal(pl_res[llik_ret_str()], pl_res2[llik_ret_str()])


def test_output_parameter():
    pl_res2 = ebnm_point_laplace(x, s, output=("fitted_g",))
    expect_identical(tuple(pl_res2), ("fitted_g",))


def test_can_fix_g_with_one_component():
    g_init1 = dict(pi=1, scale=true_scale, mean=true_mean)

    np.random.seed(0)
    pl_res1 = ebnm_point_laplace(x, s, g_init=g_init1, fix_g=True)

    g_init2 = dict(pi=(0, 1), scale=(0, true_scale), mean=rep(true_mean, 2))

    np.random.seed(0)
    pl_res2 = ebnm_point_laplace(x, s, g_init=g_init2, fix_g=True)

    expect_equal(pl_res1["log_likelihood"], pl_res2["log_likelihood"])


def test_very_large_observations_give_reasonable_results():
    pl_res_small = ebnm_point_laplace(x, s, mode="estimate")
    pl_res_lg = ebnm_point_laplace(x, s, mode="estimate")

    expect_equal(
        pl_res_small[g_ret_str()]["pi"][0], pl_res_lg[g_ret_str()]["pi"][0], tolerance=1e-3
    )
    expect_equal(
        pl_res_small[g_ret_str()]["scale"][1], pl_res_lg[g_ret_str()]["scale"][1], tolerance=1e-3
    )
    expect_equal(
        pl_res_small[g_ret_str()]["mean"][0], pl_res_lg[g_ret_str()]["mean"][0], tolerance=1e-3
    )


def test_very_small_observations_give_reasonable_results():
    scl = 1e-8

    pl_res_ = ebnm_point_laplace(x, s, mode="estimate")
    pl_res_lg = ebnm_point_laplace(scl * x, scl * s, mode="estimate")

    expect_equal(pl_res_[g_ret_str()]["pi"][0], pl_res_lg[g_ret_str()]["pi"][0])
    expect_equal(scl * pl_res_[g_ret_str()]["scale"][1], pl_res_lg[g_ret_str()]["scale"][1])
    expect_equal(scl * pl_res_[g_ret_str()]["mean"][0], pl_res_lg[g_ret_str()]["mean"][0])
