import warnings

import numpy as np
from numpy import exp, log

from .r_utils import pmax, pmin, rep, stop
from .r_utils.stat import dnorm, pnorm


def my_etruncnorm(a, b, mean=0, sd=1):
    a_is_scalar = np.isscalar(a)
    b_is_scalar = np.isscalar(b)

    if a_is_scalar:
        a = np.array([a])
    if b_is_scalar:
        b = np.array([b])

    do_truncnorm_argchecks(a, b)

    alpha = (a - mean) / sd
    beta = (b - mean) / sd

    flip = ((alpha > 0) & (beta > 0)) | (beta > abs(alpha))
    flip[np.isnan(flip)] = False
    orig_alpha = alpha.copy()
    alpha[flip] = -beta[flip]
    beta[flip] = -orig_alpha[flip]

    dnorm_diff = logscale_sub(dnorm(beta, log=True), dnorm(alpha, log=True))
    pnorm_diff = logscale_sub(pnorm(beta, log_p=True), pnorm(alpha, log_p=True))
    scaled_res = -exp(dnorm_diff - pnorm_diff)

    endpts_equal = np.isinf(pnorm_diff)
    scaled_res[endpts_equal] = (alpha[endpts_equal] + beta[endpts_equal]) / 2
    lower_bd = pmax(beta + 1 / beta, (alpha + beta) / 2)
    bad_idx = ~np.isnan(beta) & (beta < 0) & ((scaled_res < lower_bd) | (scaled_res > beta))
    scaled_res[bad_idx] = lower_bd[bad_idx]

    scaled_res[flip] = -scaled_res[flip]

    res = mean + sd * scaled_res

    if np.any(sd == 0):
        a = rep(a, length_out=len(res))
        b = rep(b, length_out=len(res))
        mean = rep(mean, length_out=len(res))

        sd_zero = sd == 0
        res[sd_zero & (b <= mean)] = b[sd_zero & (b <= mean)]
        res[sd_zero & (a >= mean)] = a[sd_zero & (a >= mean)]
        res[sd_zero & (a < mean) & (b > mean)] = mean[sd_zero & (a < mean) & (b > mean)]

    if len(res) == 1 and a_is_scalar and b_is_scalar:
        return res.item()

    return res


def my_e2truncnorm(a, b, mean=0, sd=1):
    a_is_scalar = np.isscalar(a)
    b_is_scalar = np.isscalar(b)

    if a_is_scalar:
        a = np.array([a])
    if b_is_scalar:
        b = np.array([b])

    do_truncnorm_argchecks(a, b)

    alpha = (a - mean) / sd
    beta = (b - mean) / sd

    flip = (alpha > 0) & (beta > 0)
    flip[np.isnan(flip)] = False
    orig_alpha = alpha.copy()
    alpha[flip] = -beta[flip]
    beta[flip] = -orig_alpha[flip]
    if np.any(mean != 0):
        mean = rep(mean, length_out=len(alpha))
        mean[flip] = -mean[flip]

    pnorm_diff = logscale_sub(pnorm(beta, log_p=True), pnorm(alpha, log_p=True))
    alpha_frac = alpha * exp(dnorm(alpha, log=True) - pnorm_diff)
    beta_frac = beta * exp(dnorm(beta, log=True) - pnorm_diff)

    scaled_res = np.ones_like(alpha)

    scaled_res[np.isnan(flip)] = flip[np.isnan(flip)]

    alpha_idx = np.isfinite(alpha)
    scaled_res[alpha_idx] = 1 + alpha_frac[alpha_idx]
    beta_idx = np.isfinite(beta)
    scaled_res[beta_idx] = scaled_res[beta_idx] - beta_frac[beta_idx]

    # Handle approximately equal endpoints_
    endpts_equal = np.isinf(pnorm_diff)
    scaled_res[endpts_equal] = (alpha[endpts_equal] + beta[endpts_equal]) ** 2 / 4

    upper_bd1 = beta ** 2 + 2 * (1 + 1 / beta ** 2)
    upper_bd2 = (alpha ** 2 + alpha * beta + beta ** 2) / 3
    upper_bd = pmin(upper_bd1, upper_bd2)
    bad_idx = ~np.isnan(beta) & (beta < 0) & ((scaled_res < beta ** 2) | (scaled_res > upper_bd))
    scaled_res[bad_idx] = upper_bd[bad_idx]

    res = mean ** 2 + 2 * mean * sd * my_etruncnorm(alpha, beta) + sd ** 2 * scaled_res

    if np.any(sd == 0):
        a = rep(a, length_out=len(res))
        b = rep(b, length_out=len(res))
        mean = rep(mean, length_out=len(res))

        sd_zero = sd == 0
        res[sd_zero & (b <= mean)] = b[(sd_zero & b) <= mean] ** 2
        res[sd_zero & (a >= mean)] = a[sd_zero & (a >= mean)] ** 2
        res[sd_zero & (a < mean) & (b > mean)] = mean[sd_zero & (a < mean) & (b > mean)] ** 2

    if len(res) == 1 and a_is_scalar and b_is_scalar:
        return res.item()

    return res


def do_truncnorm_argchecks(a, b):
    if len(a) != len(b):
        stop("truncnorm functions require that a and b have the same length.")
    if np.any(b < a):
        stop("truncnorm functions require that a <= b.")


def logscale_sub(logx, logy):
    diff = logx - logy
    if np.any(diff < 0):
        bad_idx = diff < 0
        bad_idx[np.isnan(bad_idx)] = False
        logx[bad_idx] = logy[bad_idx]
        warnings.warn(
            f"logscale_sub encountered negative value(s) of logx - logy (min: {min(diff[bad_idx])})"
        )

    scale_by = logx
    scale_by[np.isinf(scale_by)] = 0
    return log(exp(logx - scale_by) - exp(logy - scale_by)) + scale_by
