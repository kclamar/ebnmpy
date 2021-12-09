import warnings

import numpy as np

from .output import as_ebnm, lfsr_arg_str, output_all, output_default
from .point_exponential import (
    pe_checkg,
    pe_initpar,
    pe_nllik,
    pe_partog,
    pe_postcomp,
    pe_postsamp,
    pe_precomp,
    pe_scalepar,
    pe_summres,
)
from .point_laplace import (
    pl_checkg,
    pl_initpar,
    pl_nllik,
    pl_partog,
    pl_postcomp,
    pl_postsamp,
    pl_precomp,
    pl_scalepar,
    pl_summres,
)
from .point_normal import (
    pn_checkg,
    pn_initpar,
    pn_nllik,
    pn_partog,
    pn_postcomp,
    pn_postsamp,
    pn_precomp,
    pn_scalepar,
    pn_summres,
)
from .r_utils import length, stop
from .workhorse_parametric import parametric_workhorse


def ebnm(
    x,
    s=1,
    *,
    prior_family,
    mode=0,
    scale="estimate",
    g_init=None,
    fix_g=False,
    output=output_default(),
    optmethod=None,
    control=None,
):
    return ebnm_workhorse(
        x=x,
        s=s,
        mode=mode,
        scale=scale,
        g_init=g_init,
        fix_g=fix_g,
        output=output,
        optmethod=optmethod,
        control=control,
        prior_family=prior_family,
    )


def ebnm_workhorse(
    x,
    s,
    mode,
    scale,
    g_init,
    fix_g,
    output,
    optmethod,
    control,
    prior_family,
):
    if np.isscalar(x):
        x = np.array([x])
    if np.isscalar(s):
        s = np.array([s])

    x = x.copy()
    s = s.copy()

    check_args(x, s, g_init, fix_g, output, mode)
    mode = handle_mode_parameter(mode)
    scale = handle_scale_parameter(scale)

    if control is None:
        control = dict()

    if lfsr_arg_str() in output and mode != 0:
        warnings.warn(
            "Since they're not well defined for nonzero modes, local false sign rates won't be "
            "returned. "
        )
        output = tuple(i for i in output if i != lfsr_arg_str())

    if prior_family == "point_normal":
        retlist = parametric_workhorse(
            x=x,
            s=s,
            mode=mode,
            scale=scale,
            pointmass=True,
            g_init=g_init,
            fix_g=fix_g,
            output=output,
            optmethod=optmethod,
            control=control,
            checkg_fn=pn_checkg,
            initpar_fn=pn_initpar,
            scalepar_fn=pn_scalepar,
            precomp_fn=pn_precomp,
            nllik_fn=pn_nllik,
            postcomp_fn=pn_postcomp,
            summres_fn=pn_summres,
            partog_fn=pn_partog,
            postsamp_fn=pn_postsamp,
        )
    elif prior_family == "point_laplace":
        retlist = parametric_workhorse(
            x=x,
            s=s,
            mode=mode,
            scale=scale,
            pointmass=True,
            g_init=g_init,
            fix_g=fix_g,
            output=output,
            optmethod=optmethod,
            control=control,
            checkg_fn=pl_checkg,
            initpar_fn=pl_initpar,
            scalepar_fn=pl_scalepar,
            precomp_fn=pl_precomp,
            nllik_fn=pl_nllik,
            postcomp_fn=pl_postcomp,
            summres_fn=pl_summres,
            partog_fn=pl_partog,
            postsamp_fn=pl_postsamp,
        )
    elif prior_family == "point_exponential":
        retlist = parametric_workhorse(
            x=x,
            s=s,
            mode=mode,
            scale=scale,
            pointmass=True,
            g_init=g_init,
            fix_g=fix_g,
            output=output,
            optmethod=optmethod,
            control=control,
            checkg_fn=pe_checkg,
            initpar_fn=pe_initpar,
            scalepar_fn=pe_scalepar,
            precomp_fn=pe_precomp,
            nllik_fn=pe_nllik,
            postcomp_fn=pe_postcomp,
            summres_fn=pe_summres,
            partog_fn=pe_partog,
            postsamp_fn=pe_postsamp,
        )
    elif prior_family == "normal":
        retlist = parametric_workhorse(
            x=x,
            s=s,
            mode=mode,
            scale=scale,
            pointmass=False,
            g_init=g_init,
            fix_g=fix_g,
            output=output,
            optmethod=optmethod,
            control=control,
            checkg_fn=pn_checkg,
            initpar_fn=pn_initpar,
            scalepar_fn=pn_scalepar,
            precomp_fn=pn_precomp,
            nllik_fn=pn_nllik,
            postcomp_fn=pn_postcomp,
            summres_fn=pn_summres,
            partog_fn=pn_partog,
            postsamp_fn=pn_postsamp,
        )
    elif prior_family == "horseshoe":
        # TODO
        raise NotImplementedError
    elif prior_family == "normal_scale_mixture":
        # TODO
        raise NotImplementedError
    elif prior_family == "unimodal":
        # TODO
        raise NotImplementedError
    elif prior_family == "unimodal_symmetric":
        # TODO
        raise NotImplementedError
    elif prior_family == "unimodal_nonnegative":
        # TODO
        raise NotImplementedError
    elif prior_family == "unimodal_nonpositive":
        # TODO
        raise NotImplementedError
    elif prior_family == "ash":
        # TODO
        raise NotImplementedError
    elif prior_family == "npmle" and optmethod == "REBayes":
        # TODO
        raise NotImplementedError
    elif prior_family == "npmle":
        # TODO
        raise NotImplementedError
    elif prior_family == "deconvolver":
        # TODO
        raise NotImplementedError
    else:
        raise NotImplementedError

    return as_ebnm(retlist)


def check_args(x, s, g_init, fix_g, output, mode):
    if length(s) not in (1, len(x)):
        stop("Argument 's' must have either length 1 or the same length as argument 'x'.")

    if np.any(np.isnan(x)):
        stop("Missing observations are not allowed.")

    if np.any(np.isnan(s)):
        stop("Missing standard errors are not allowed.")

    if np.any(s <= 0):
        stop("Standard errors must be positive (and nonzero).")

    if np.any(np.isinf(s)):
        stop("Standard errors cannot be infinite.")

    if fix_g and g_init is None:
        stop("If g is fixed, then an initial g must be provided.")

    if not np.all(i in output_all() for i in output):
        stop("Invalid argument to output. See function output_all() for a list of valid outputs.")


def handle_mode_parameter(mode):
    if mode == "estimate":
        pass
    elif not (isinstance(mode, (int, float)) and np.isfinite(mode)):
        stop("Argument 'mode' must be either 'estimate' or a numeric value.")
    return mode


def handle_scale_parameter(scale):
    if scale == "estimate":
        pass
    elif not (isinstance(scale, (int, float)) and np.isfinite(scale)):
        stop("Argument 'scale' must be either 'estimate' or numeric.")
    return scale
