import warnings

import numpy as np
from scipy.optimize import minimize

from .opt_control_defaults import lbfgsb_control_defaults
from .output import (
    add_g_to_retlist,
    add_llik_to_retlist,
    add_posterior_to_retlist,
    add_sampler_to_retlist,
    g_in_output,
    llik_in_output,
    posterior_in_output,
    sampler_in_output,
)
from .rutils import stop


def handle_optmethod_parameter(optmethod, fix_par):
    if optmethod == "lbfgsb":
        return dict(fn="lbfgsb", use_grad=True, use_hess=False)
    if optmethod == "nograd_lbfgsb":
        return dict(fn="lbfgsb", use_grad=False, use_hess=False)

    raise NotImplementedError


def mle_parametric(
    x,
    s,
    par_init,
    fix_par,
    scalepar_fn,
    precomp_fn,
    nllik_fn,
    postcomp_fn,
    optmethod,
    control,
    use_grad,
    use_hess,
):
    scale_factor = 1 / np.median(s[s > 0])
    x = x * scale_factor
    s = s * scale_factor

    par_init = scalepar_fn(par=par_init, scale_factor=scale_factor)

    precomp = precomp_fn(x=x, s=s, par_init=par_init, fix_par=fix_par)

    fn_params = dict(precomp, x=x, s=s, par_init=par_init, fix_par=fix_par)

    p = np.array(list(par_init.values()))[~np.array(fix_par)]

    if (not fix_par[1]) and np.isinf(p[0]):
        p[0] = np.sign(p[0]) * np.log(x)

    if all(fix_par):
        raise NotImplementedError
    elif optmethod == "lbfgsb":
        control = dict(lbfgsb_control_defaults(), **control)

        def fn(par, kwargs):
            return nllik_fn(par, calc_grad=False, calc_hess=False, **kwargs)

        if use_grad:

            def gr(par, kwargs):
                return nllik_fn(par, calc_grad=True, calc_hess=False, **kwargs)

        else:
            gr = None

        optres = minimize(
            fun=fn,
            x0=p,
            jac=gr,
            args=(fn_params,),
            options=control,
            method="L-BFGS-B",
        )
        optpar = optres.x
        optval = optres.fun
    else:
        raise NotImplementedError

    retpar = par_init

    retpar_values = np.array(list(retpar.values()))
    retpar_values[~fix_par] = optpar
    retpar = dict(zip(list(retpar), retpar_values))

    retpar = scalepar_fn(par=retpar, scale_factor=1 / scale_factor)
    optval = optval - sum(np.isfinite(x) * np.log(scale_factor))

    retlist = postcomp_fn(
        optpar=retpar,
        optval=optval,
        x=x,
        s=s,
        par_init=par_init,
        fix_par=fix_par,
        scale_factor=scale_factor,
        **precomp,
    )

    return retlist


def parametric_workhorse(
    x,
    s,
    mode,
    scale,
    pointmass,
    g_init,
    fix_g,
    output,
    optmethod,
    control,
    checkg_fn,
    initpar_fn,
    scalepar_fn,
    precomp_fn,
    nllik_fn,
    postcomp_fn,
    summres_fn,
    partog_fn,
    postsamp_fn,
):
    checkg_fn(
        g_init=g_init,
        fix_g=fix_g,
        mode=mode,
        scale=scale,
        pointmass=pointmass,
    )

    par_init = initpar_fn(g_init=g_init, mode=mode, scale=scale, pointmass=pointmass, x=x, s=s)

    if fix_g:
        fix_par = np.array([True, True, True])
    else:
        fix_par = np.array([not pointmass, scale != "estimate", mode != "estimate"])

    optmethod = handle_optmethod_parameter(optmethod, fix_par)

    x_optset = x
    s_optset = s

    if np.any(np.isinf(s)):
        x_optset = x[np.isfinite(s)]
        s_optset = s[np.isfinite(s)]

    optres = mle_parametric(
        x=x_optset,
        s=s_optset,
        par_init=par_init,
        fix_par=fix_par,
        scalepar_fn=scalepar_fn,
        precomp_fn=precomp_fn,
        nllik_fn=nllik_fn,
        postcomp_fn=postcomp_fn,
        optmethod=optmethod["fn"],
        control=control,
        use_grad=optmethod["use_grad"],
        use_hess=optmethod["use_hess"],
    )

    retlist = dict()
    if posterior_in_output(output):
        posterior = summres_fn(x=x, s=s, optpar=optres["par"], output=output)
        retlist = add_posterior_to_retlist(retlist, posterior, output)

    if g_in_output(output):
        fitted_g = partog_fn(par=optres["par"])
        retlist = add_g_to_retlist(retlist, fitted_g)

    if llik_in_output(output):
        loglik = optres["val"]
        retlist = add_llik_to_retlist(retlist, loglik)

    if sampler_in_output(output):

        def post_sampler(nsamp):
            return postsamp_fn(x, s, optres["par"], nsamp)

        retlist = add_sampler_to_retlist(retlist, post_sampler)

    return retlist


def check_g_init(
    g_init,
    fix_g,
    mode,
    scale,
    pointmass,
    class_name,
    scale_name,
    mode_name="mean",
):
    if g_init is not None:
        ncomp = len(np.array([g_init["pi"]]).ravel())
        if not (ncomp == 1 or (pointmass and ncomp == 2)):
            stop("g_init does not have the correct number of components.")
        if ncomp == 2 and g_init[scale_name][0] != 0:
            stop("The first component of g_init must be a point mass.")

        if fix_g and (mode is not None or scale is not None):
            warnings.warn("mode and scale parameters are ignored when g is fixed.")

        if not fix_g:
            if mode is not None and mode != "estimate" and not np.all(g_init[mode_name] == mode):
                stop("If mode is fixed and g_init is supplied, they must agree.")
            g_scale = g_init[scale_name][ncomp - 1]
            if scale is not None and scale != "estimate" and not np.all(g_scale == scale):
                stop("If scale is fixed and g_init is supplied, they must agree.")
