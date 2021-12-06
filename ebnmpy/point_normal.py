import numpy as np
from numpy import exp, inf, log, mean, sqrt
from scipy.stats import binom, norm

from .output import lfsr_in_output, result_in_output
from .r_utils import length, stop
from .workhorse_parametric import check_g_init


def pn_checkg(g_init, fix_g, mode, scale, pointmass):
    return check_g_init(
        g_init=g_init,
        fix_g=fix_g,
        mode=mode,
        scale=scale,
        pointmass=pointmass,
        class_name="normalmix",
        scale_name="sd",
    )


def pn_initpar(g_init, mode, scale, pointmass, x, s):
    if g_init is not None and length(g_init["pi"]) == 1:
        par = dict(alpha=-inf, beta=2 * log(g_init["sd"]), mu=g_init["mean"])
    elif g_init is not None and length(g_init["pi"]) == 2:
        par = dict(
            alpha=-log(1 / g_init["pi"][0] - 1),
            beta=2 * log(g_init["sd"][1]),
            mu=g_init["mean"][0],
        )
    else:
        par = dict()

        if not pointmass:
            par["alpha"] = -inf
        else:
            par["alpha"] = 0

        if scale != "estimate":
            if length(scale) != 1:
                stop("Argument 'scale' must be either 'estimate' or a scalar.")
            par["beta"] = 2 * log(scale)
        else:
            par["beta"] = log(mean(x ** 2))  # default

        if mode != "estimate":
            par["mu"] = mode
        else:
            par["mu"] = mean(x)  # default

    return par


def pn_scalepar(par, scale_factor):
    if par["beta"] is not None:
        par["beta"] = par["beta"] + 2 * log(scale_factor)

    if par["mu"] is not None:
        par["mu"] = scale_factor * par["mu"]

    return par


def pn_precomp(x, s, par_init, fix_par):
    fix_mu = fix_par[2]

    if not fix_mu and np.any(s == 0):
        stop("The mode cannot be estimated if any SE is zero (the gradient does not exist).")

    if np.any(s == 0):
        which_s0 = np.equal(s, 0)
        which_x_nz = np.not_equal(x[which_s0], par_init["mu"])
        n0 = sum(which_s0) - sum(which_x_nz)
        n1 = sum(which_x_nz)
        sum1 = sum((x[which_s0[which_x_nz]] - par_init["mu"]) ** 2)
        x = x[~which_s0]
        s = s[~which_s0]
    else:
        n0 = 0
        n1 = 0
        sum1 = 0

    n2 = len(x)

    s2 = s ** 2

    if fix_mu:
        z = (x - par_init["mu"]) ** 2 / s2
        sum_z = sum(z)
    else:
        z = None
        sum_z = None

    return dict(n0=n0, n1=n1, sum1=sum1, n2=n2, s2=s2, z=z, sum_z=sum_z)


def pn_nllik(
    par,
    x,
    s,
    par_init,
    fix_par,
    n0,
    n1,
    sum1,
    n2,
    s2,
    z,
    sum_z,
    calc_grad,
    calc_hess,
):
    fix_pi0, fix_s2, fix_mu = fix_par

    i = 0
    if fix_pi0:
        alpha = par_init["alpha"]
    else:
        alpha = par[i]
        i = i + 1

    if fix_s2:
        beta = par_init["beta"]
    else:
        beta = par[i]
        i = i + 1

    if fix_mu:
        mu = par_init["mu"]
    else:
        mu = par[i]
        z = (x - mu) ** 2 / s2
        sum_z = sum(z)

    logist_alpha = 1 / (1 + exp(-alpha))  # scalar
    logist_nalpha = 1 / (1 + exp(alpha))

    logist_beta = 1 / (1 + s2 * exp(-beta))  # scalar or vector
    logist_nbeta = 1 / (1 + exp(beta) / s2)

    y = 0.5 * (z * logist_beta + log(logist_nbeta))  # vector

    # Negative log likelihood.
    C = np.maximum(alpha, y)
    if n0 == 0 or logist_alpha == 0:
        nllik = 0
    else:
        nllik = -n0 * log(logist_alpha)

    nllik = nllik - (n1 + n2) * (log(logist_nalpha))
    nllik = nllik + 0.5 * (n1 * beta + sum1 * exp(-beta) + sum_z)
    nllik = nllik - sum(log(exp(y - C) + exp(alpha - C)) + C)

    if calc_grad or calc_hess:
        dlogist_beta = logist_beta * logist_nbeta

        logist_y = 1 / (1 + exp(alpha - y))  # vector
        logist_ny = 1 / (1 + exp(y - alpha))

        # Gradient.
        grad = np.zeros(len(par))
        i = 0
        if not fix_pi0:
            grad[i] = -n0 * logist_nalpha + (n1 + n2) * logist_alpha - sum(logist_ny)
            i = i + 1

        if not fix_s2:
            dy_dbeta = 0.5 * (z * dlogist_beta - logist_beta)
            grad[i] = 0.5 * (n1 - sum1 * exp(-beta)) - sum(logist_y * dy_dbeta)
            i = i + 1

        if not fix_mu:
            dy_dmu = (mu - x) * logist_beta / s2
            grad[i] = sum((mu - x) / s2) - sum(logist_y * dy_dmu)

        return grad

    if calc_hess:
        # TODO
        raise NotImplementedError

    return nllik


def pn_postcomp(
    optpar,
    optval,
    x,
    s,
    par_init,
    fix_par,
    scale_factor,
    n0,
    n1,
    sum1,
    n2,
    s2,
    z,
    sum_z,
):
    llik = pn_llik_from_optval(optval, n1, n2, s2)
    retlist = dict(par=optpar, val=llik)

    fix_pi0 = fix_par[0]
    fix_mu = fix_par[2]
    if not fix_pi0 and fix_mu:
        pi0_llik = sum(-0.5 * log(2 * np.pi * s ** 2) - 0.5 * (x - par_init["mu"]) ** 2 / s ** 2)
        pi0_llik = pi0_llik + sum(np.isfinite(x)) * log(scale_factor)
        if pi0_llik > llik:
            retlist["par"]["alpha"] = inf
            retlist["par"]["beta"] = 0
            retlist["val"] = pi0_llik

    return retlist


def pn_llik_from_optval(optval, n1, n2, s2):
    if len(s2) == 1:
        sum_log_s2 = n2 * log(s2)
    else:
        sum_log_s2 = sum(log(s2))

    return -optval - 0.5 * ((n1 + n2) * log(2 * np.pi) + sum_log_s2)


def pn_summres(x, s, optpar, output):
    w = 1 - 1 / (exp(-optpar["alpha"]) + 1)
    a = exp(-optpar["beta"])
    mu = optpar["mu"]

    return pn_summres_untransformed(x, s, w, a, mu, output)


def pn_summres_untransformed(x, s, w, a, mu, output):
    wpost = wpost_normal(x, s, w, a, mu)
    pmean_cond = pmean_cond_normal(x, s, a, mu)
    pvar_cond = pvar_cond_normal(s, a)

    posterior = dict()

    if result_in_output(output):
        posterior["mean"] = wpost * pmean_cond + (1 - wpost) * mu
        posterior["mean2"] = wpost * (pmean_cond ** 2 + pvar_cond) + (1 - wpost) * mu ** 2
        posterior["sd"] = sqrt(np.maximum(0, posterior["mean2"] - posterior["mean"] ** 2))

        if lfsr_in_output(output):
            posterior["lfsr"] = (1 - wpost) + wpost * norm.cdf(0, abs(pmean_cond), sqrt(pvar_cond))

    return posterior


def wpost_normal(x, s, w, a, mu):
    if w == 0:
        return np.zeros(len(x))

    if w == 1:
        return np.ones(len(x))

    llik_diff = 0.5 * log(1 + 1 / (a * s ** 2))
    llik_diff = llik_diff - 0.5 * (x - mu) ** 2 / (s ** 2 * (a * s ** 2 + 1))
    wpost = w / (w + (1 - w) * exp(llik_diff))

    if any(s == 0):
        wpost[(s == 0) & (x == mu)] = 0
        wpost[(s == 0) & (x != mu)] = 1

    if any(np.isinf(s)):
        wpost[np.isinf(s)] = w

    return wpost


def pmean_cond_normal(x, s, a, mu):
    pm = (x + s ** 2 * a * mu) / (1 + s ** 2 * a)

    if any(np.isinf(s)):
        pm[np.isinf(s)] = mu

    return pm


def pvar_cond_normal(s, a):
    pvar_cond = s ** 2 / (1 + s ** 2 * a)

    if any(np.isinf(s)):
        pvar_cond[np.isinf(s)] = 1 / a

    return pvar_cond


def pn_partog(par):
    pi0 = 1 / (exp(-par["alpha"]) + 1)
    sd = exp(par["beta"] / 2)
    mean = par["mu"]

    if pi0 == 0:
        g = dict(pi=1, mean=mean, sd=sd)
    else:
        g = dict(pi=(pi0, 1 - pi0), mean=(mean, mean), sd=(0, sd))

    return g


def pn_postsamp(x, s, optpar, nsamp):
    w = 1 - 1 / (exp(-optpar["alpha"]) + 1)
    a = exp(-optpar["beta"])
    mu = optpar["mu"]

    return pn_postsamp_untransformed(x, s, w, a, mu, nsamp)


def pn_postsamp_untransformed(x, s, w, a, mu, nsamp):
    wpost = wpost_normal(x, s, w, a, mu)
    pmean_cond = pmean_cond_normal(x, s, a, mu)
    pvar_cond = pvar_cond_normal(s, a)

    nobs = len(x)
    is_nonnull = binom.rvs(n=1, p=np.repeat(wpost, nsamp), size=nsamp * nobs)
    samp = is_nonnull * norm.rvs(
        loc=np.repeat(pmean_cond, nsamp),
        scale=np.repeat(sqrt(pvar_cond), nsamp),
        size=nsamp * nobs,
    )
    samp = samp + (1 - is_nonnull) * mu

    return samp.reshape(nsamp, -1)
