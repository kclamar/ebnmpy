import numpy as np
from numpy import exp, inf, log, mean, sqrt
from scipy.stats import bernoulli

from .ashr import my_e2truncnorm, my_etruncnorm
from .output import result_in_output
from .point_laplace import logscale_add
from .r_utils import length, numeric, pmax, rep, stop, unlist
from .r_utils.stats import dnorm, pnorm, rtruncnorm
from .workhorse_parametric import check_g_init


def gammamix(pi, shape, scale, shift=None):
    if shift is None:
        shift = rep(0, len(pi))

    return dict(pi=pi, shape=shape, scale=scale, shift=shift)


def pe_checkg(g_init, fix_g, mode, scale, pointmass):
    check_g_init(
        g_init=g_init,
        fix_g=fix_g,
        mode=mode,
        scale=scale,
        pointmass=pointmass,
        class_name="gammamix",
        scale_name="scale",
        mode_name="shift",
    )

    if g_init is not None and not np.all(g_init["shape"] == rep(1, length(g_init["shape"]))):
        stop("g_init must be of class gammamix with shape parameter = 1")


def pe_initpar(g_init, mode, scale, pointmass, x, s):
    if g_init is not None and length(g_init["pi"]) == 1:
        par = dict(alpha=inf, beta=-log(g_init["scale"]), mu=g_init["shift"])
    elif g_init is not None and length(g_init["pi"]) == 2:
        par = dict(
            alpha=log(1 / g_init["pi"][0] - 1) if g_init["pi"][0] != 0 else inf,
            beta=-log(g_init["scale"][1]),
            mu=g_init["shift"][0],
        )
    else:
        par = dict()
        if not pointmass:
            par["alpha"] = inf
        else:
            par["alpha"] = 0
        if scale != "estimate":
            if length(scale) != 1:
                stop("Argument 'scale' must be either 'estimate' or a scalar.")
            par["beta"] = -log(scale)
        else:
            par["beta"] = -0.5 * log(mean(x ** 2) / 2)
        if mode != "estimate":
            par["mu"] = mode
        else:
            par["mu"] = min(x)

    return par


def pe_scalepar(par, scale_factor):
    if par["beta"] is not None:
        par["beta"] -= log(scale_factor)
    if par["mu"] is not None:
        par["mu"] = scale_factor * par["mu"]

    return par


def pe_precomp(x, s, par_init, fix_par):
    fix_mu = fix_par[2]

    if not fix_mu and np.any(s == 0):
        stop("The mode cannot be estimated if any SE is zero (the gradient does not exist).")

    return dict()


def pe_nllik(par, x, s, par_init, fix_par, calc_grad, calc_hess):
    fix_pi0, fix_a, fix_mu = fix_par

    p = unlist(par_init)
    p[~np.array(fix_par)] = par

    w = 1 - 1 / (1 + exp(p[0]))
    a = exp(p[1])
    mu = p[2]

    lf = -0.5 * log(2 * np.pi * s ** 2) - 0.5 * (x - mu) ** 2 / s ** 2

    xright = (x - mu) / s - s * a
    lpnormright = pnorm(xright, log_p=True)
    lg = log(a) + s ** 2 * a ** 2 / 2 - a * (x - mu) + lpnormright

    llik = logscale_add(log(1 - w) + lf, log(w) + lg)
    nllik = -sum(llik)

    if calc_grad or calc_hess:
        grad = numeric(length(par))
        i = 0
        if not fix_pi0:
            f = exp(lf - llik)
            g = exp(lg - llik)
            dnllik_dw = f - g
            dw_dalpha = w * (1 - w)
            dnllik_dalpha = dnllik_dw * dw_dalpha

            grad[i] = sum(dnllik_dalpha)

            i += 1

        if not fix_a or not fix_mu:
            dlogpnorm_right = exp(-log(2 * np.pi) / 2 - xright ** 2 / 2 - lpnormright)

        if not fix_a:
            dg_da = exp(lg - llik) * (1 / a + a * s ** 2 - (x - mu) - s * dlogpnorm_right)
            dnllik_da = -w * dg_da
            da_dbeta = a
            dnllik_dbeta = dnllik_da * da_dbeta

            grad[i] = sum(dnllik_dbeta)
            i += 1

        if not fix_mu:
            df_dmu = exp(lf - llik) * ((x - mu) / s ** 2)
            dg_dmu = exp(lg - llik) * (a - dlogpnorm_right / s)
            dnllik_dmu = -(1 - w) * df_dmu - w * dg_dmu

            grad[i] = sum(dnllik_dmu)

        return grad

    if calc_hess:
        # TODO
        raise NotImplementedError

    return nllik


def pe_postcomp(optpar, optval, x, s, par_init, fix_par, scale_factor):
    llik = -optval
    retlist = dict(par=optpar, val=llik)

    fix_pi0 = fix_par[0]
    fix_mu = fix_par[2]
    if not fix_pi0 and fix_mu:
        pi0_llik = sum(-0.5 * log(2 * np.pi * s ** 2) - 0.5 * (x - par_init["mu"]) ** 2 / s ** 2)
        pi0_llik += sum(np.isfinite(x)) * log(scale_factor)
        if pi0_llik > llik:
            retlist["par"]["alpha"] = -inf
            retlist["par"]["beta"] = 0
            retlist["val"] = pi0_llik

    return retlist


def pe_summres(x, s, optpar, output):
    w = 1 - 1 / (exp(optpar["alpha"]) + 1)
    a = exp(optpar["beta"])
    mu = optpar["mu"]

    return pe_summres_untransformed(x, s, w, a, mu, output)


def pe_summres_untransformed(x, s, w, a, mu, output):
    x = x - mu

    wpost = wpost_exp(x, s, w, a)

    post = dict()

    if result_in_output(output):
        post["mean"] = wpost * my_etruncnorm(0, inf, x - s ** 2 * a, s)
        post["mean2"] = wpost * my_e2truncnorm(0, inf, x - s ** 2 * a, s)

        if np.any(np.isinf(s)):
            post["mean"][np.isinf(s)] = w / 2
            post["mean2"][np.isinf(s)] = 2 * w / a * 2

        post["sd"] = sqrt(pmax(0, post["mean2"] - post["mean"] ** 2))

        post["mean2"] += mu ** 2 + 2 * mu * post["mean"]
        post["mean"] += mu

    if "lfsr" in output:
        post["lfsr"] = 1 - wpost
        if np.any(np.isinf(s)):
            post["lfsr"][np.isinf(s)] = 1 - w

    return post


def wpost_exp(x, s, w, a):
    if w == 0:
        return rep(0, length(x))

    if w == 1:
        return rep(1, length(x))

    lf = dnorm(x, 0, s, log=True)
    lg = log(a) + s ** 2 * a ** 2 / 2 - a * x + pnorm(x / s - s * a, log_p=True)
    wpost = w / (w + (1 - w) * exp(lf - lg))

    return wpost


def pe_partog(par):
    pi0 = 1 / (exp(par["alpha"]) + 1)
    scale = exp(-par["beta"])
    mode = par["mu"]

    if pi0 == 0:
        g = gammamix(pi=1, shape=1, scale=scale, shift=mode)
    else:
        g = gammamix(pi=(pi0, 1 - pi0), shape=(1, 1), scale=(0, scale), shift=rep(mode, 2))

    return g


def pe_postsamp(x, s, optpar, nsamp):
    w = 1 - 1 / (exp(optpar["alpha"]) + 1)
    a = exp(optpar["beta"])
    mu = optpar["mu"]

    return pe_postsamp_untransformed(x, s, w, a, mu, nsamp)


def pe_postsamp_untransformed(x, s, w, a, mu, nsamp):
    x = x - mu

    wpost = wpost_exp(x, s, w, a)

    nobs = length(wpost)

    is_nonnull = bernoulli.rvs(wpost, size=(nsamp, nobs)) != 0

    if length(s) == 1:
        s = rep(s, nobs)

    samp = np.zeros((nsamp, nobs))
    positive_samp = np.array(
        [rtruncnorm(nsamp, 0, inf, mi, si) for mi, si in zip(x - s ** 2 * a, s)]
    ).T

    samp[is_nonnull] = positive_samp[is_nonnull]
    samp = samp + mu

    return samp
