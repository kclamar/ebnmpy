import numpy as np
from numpy import exp, inf, log, mean, sqrt
from scipy.stats import bernoulli

from .ashr import my_e2truncnorm, my_etruncnorm
from .output import result_in_output
from .r_utils import length, numeric, pmax, pmin, rep, stop, unlist
from .r_utils.stats import dnorm, pnorm, rtruncnorm
from .workhorse_parametric import check_g_init


def laplacemix(pi, mean, scale):
    return dict(pi=pi, mean=mean, scale=scale)


def pl_checkg(g_init, fix_g, mode, scale, pointmass):
    return check_g_init(
        g_init=g_init,
        fix_g=fix_g,
        mode=mode,
        scale=scale,
        pointmass=pointmass,
        class_name="laplacemix",
        scale_name="scale",
    )


def pl_initpar(g_init, mode, scale, pointmass, x, s):
    if g_init is not None and length(g_init["pi"]) == 1:
        par = dict(alpha=inf, beta=-log(g_init["scale"]), mu=g_init["mean"])
    elif g_init is not None and length(g_init["pi"]) == 2:
        par = dict(
            alpha=log(1 / g_init["pi"][0] - 1) if g_init["pi"][0] != 0 else inf,
            beta=-log(g_init["scale"][1]),
            mu=g_init["mean"][0],
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
            par["mu"] = mean(x)

    return par


def pl_scalepar(par, scale_factor):
    if par["beta"] is not None:
        par["beta"] = par["beta"] - log(scale_factor)
    if par["mu"] is not None:
        par["mu"] = scale_factor * par["mu"]

    return par


def pl_precomp(x, s, par_init, fix_par):
    fix_mu = fix_par[2]

    if not fix_mu and np.any(s == 0):
        stop("The mode cannot be estimated if any SE is zero (the gradient does not exist).")

    return dict()


def pl_nllik(par, x, s, par_init, fix_par, calc_grad, calc_hess):
    fix_pi0, fix_a, fix_mu = fix_par

    p = unlist(par_init)
    p[~np.array(fix_par)] = par

    w = 1 - 1 / (1 + exp(p[0]))
    a = exp(p[1])
    mu = p[2]

    lf = -0.5 * log(2 * np.pi * s ** 2) - 0.5 * (x - mu) ** 2 / s ** 2

    xleft = (x - mu) / s + s * a
    lpnormleft = pnorm(xleft, log_p=True, lower_tail=False)
    lgleft = log(a / 2) + s ** 2 * a ** 2 / 2 + a * (x - mu) + lpnormleft

    xright = (x - mu) / s - s * a
    lpnormright = pnorm(xright, log_p=True)
    lgright = log(a / 2) + s ** 2 * a ** 2 / 2 - a * (x - mu) + lpnormright

    lg = logscale_add(lgleft, lgright)

    llik = logscale_add(log(1 - w) + lf, log(w) + lg)
    nllik = -np.nansum(llik)

    if calc_grad or calc_hess:
        grad = numeric(len(par))
        i = 0
        if not fix_pi0:
            f = exp(lf - llik)
            g = exp(lg - llik)
            dnllik_dw = f - g
            dw_dalpha = w * (1 - w)
            dnllik_dalpha = dnllik_dw * dw_dalpha

            grad[i] = np.nansum(dnllik_dalpha)
            i += 1
        if not fix_a or not fix_mu:
            dlogpnorm_left = -exp(-log(2 * np.pi) / 2 - xleft ** 2 / 2 - lpnormleft)
            dlogpnorm_right = exp(-log(2 * np.pi) / 2 - xright ** 2 / 2 - lpnormright)
        if not fix_a:
            dgleft_da = exp(lgleft - llik) * (1 / a + a * s ** 2 + (x - mu) + s * dlogpnorm_left)
            dgright_da = exp(lgright - llik) * (1 / a + a * s ** 2 - (x - mu) - s * dlogpnorm_right)
            dg_da = dgleft_da + dgright_da
            dnllik_da = -w * dg_da
            da_dbeta = a
            dnllik_dbeta = dnllik_da * da_dbeta

            grad[i] = np.nansum(dnllik_dbeta)
            i += 1
        if not fix_mu:
            df_dmu = exp(lf - llik) * ((x - mu) / s ** 2)
            dgleft_dmu = exp(lgleft - llik) * (-a - dlogpnorm_left / s)
            dgright_dmu = exp(lgright - llik) * (a - dlogpnorm_right / s)
            dg_dmu = dgleft_dmu + dgright_dmu
            dnllik_dmu = -(1 - w) * df_dmu - w * dg_dmu

            grad[i] = np.nansum(dnllik_dmu)

        return grad

    if calc_hess:
        # TODO
        raise NotImplementedError

    return nllik


def logscale_add(log_x, log_y):
    C = pmax(log_x, log_y)
    return log(exp(log_x - C) + exp(log_y - C)) + C


def pl_postcomp(optpar, optval, x, s, par_init, fix_par, scale_factor):
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


def pl_summres(x, s, optpar, output):
    w = 1 - 1 / (exp(optpar["alpha"]) + 1)
    a = exp(optpar["beta"])
    mu = optpar["mu"]

    return pl_summres_untransformed(x, s, w, a, mu, output)


def pl_summres_untransformed(x, s, w, a, mu, output):
    x = x - mu

    wpost = wpost_laplace(x, s, w, a)
    lm = lambda_(x, s, a)

    post = dict()

    if result_in_output(output):
        post["mean"] = wpost * (
            lm * my_etruncnorm(0, inf, x - s ** 2 * a, s)
            + (1 - lm) * my_etruncnorm(-inf, 0, x + s ** 2 * a, s)
        )
        post["mean2"] = wpost * (
            lm * my_e2truncnorm(0, inf, x - s ** 2 * a, s)
            + (1 - lm) * my_e2truncnorm(-inf, 0, x + s ** 2 * a, s)
        )

        if np.any(np.isinf(s)):
            post["mean"][np.isinf(s)] = 0
            post["mean2"][np.isinf(s)] = 2 * w / a ** 2

        post["sd"] = sqrt(pmax(0, post["mean2"] - post["mean"] ** 2))

        post["mean2"] = post["mean2"] + mu ** 2 + 2 * mu * post["mean"]
        post["mean"] = post["mean"] + mu

    if "lfsr" in output:
        post["lfsr"] = (1 - wpost) + wpost * pmin(lm, 1 - lm)
        if np.any(np.isinf(s)):
            post["lfsr"][np.isinf(s)] = 1 - w / 2

    return post


def wpost_laplace(x, s, w, a):
    if w == 0:
        return np.zeros(len(x))

    if w == 1:
        return np.ones(len(x))

    lf = dnorm(x, 0, s, log=True)
    lg = logg_laplace(x, s, a)
    wpost = w / (w + (1 - w) * exp(lf - lg))

    return wpost


def logg_laplace(x, s, a):
    lg1 = -a * x + pnorm((x - s ** 2 * a) / s, log_p=True)
    lg2 = a * x + pnorm((x + s ** 2 * a) / s, log_p=True, lower_tail=False)
    lfac = pmax(lg1, lg2)
    return log(a / 2) + s ** 2 * a ** 2 / 2 + lfac + log(exp(lg1 - lfac) + exp(lg2 - lfac))


def lambda_(x, s, a):
    lm1 = -a * x + pnorm(x / s - s * a, log_p=True)
    lm2 = a * x + pnorm(x / s + s * a, log_p=True, lower_tail=False)
    lm = 1 / (1 + exp(lm2 - lm1))
    return lm


def pl_partog(par):
    pi0 = 1 / (exp(par["alpha"]) + 1)
    scale = exp(-par["beta"])
    mean = par["mu"]

    if pi0 == 0:
        g = laplacemix(pi=1, mean=mean, scale=scale)
    else:
        g = laplacemix(pi=(pi0, 1 - pi0), mean=(mean,) * 2, scale=(0, scale))

    return g


def pl_postsamp(x, s, optpar, nsamp):
    w = 1 - 1 / (exp(optpar["alpha"]) + 1)
    a = exp(optpar["beta"])
    mu = optpar["mu"]

    return pl_postsamp_untransformed(x, s, w, a, mu, nsamp)


def pl_postsamp_untransformed(x, s, w, a, mu, nsamp):
    x = x - mu
    wpost = wpost_laplace(x, s, w, a)
    lam = lambda_(x, s, a)

    nobs = len(wpost)

    is_nonnull = bernoulli.rvs(wpost, size=(nsamp, nobs)) != 0
    is_positive = bernoulli.rvs(lam, size=(nsamp, nobs)) != 0

    if len(s) == 1:
        s = rep(s, nobs)

    negative_samp = np.array(
        [rtruncnorm(nsamp, -inf, 0, mi, si) for mi, si in zip(x + s ** 2 * a, s)]
    ).T
    positive_samp = np.array(
        [rtruncnorm(nsamp, 0, inf, mi, si) for mi, si in zip(x - s ** 2 * a, s)]
    ).T

    samp = np.zeros((nsamp, nobs))
    samp[is_nonnull & is_positive] = positive_samp[is_nonnull & is_positive]
    samp[is_nonnull & ~is_positive] = negative_samp[is_nonnull & ~is_positive]

    samp = samp + mu

    return samp
