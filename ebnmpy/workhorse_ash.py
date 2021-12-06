import warnings

import numpy as np

from .ashr import ash
from .grid_selection import default_symmuni_scale
from .output import (
    add_g_to_retlist,
    add_llik_to_retlist,
    add_posterior_to_retlist,
    add_sampler_to_retlist,
    ash_output,
    g_in_output,
    lfsr_in_output,
    llik_in_output,
    posterior_in_output,
    result_in_output,
    sampler_in_output,
)
from .r_utils import asvector, stop


def ebnm_ash_workhorse(x, s, mode, scale, g_init, fix_g, output, **kwargs):
    if "mixsd" in kwargs:
        stop("Use parameter 'scale' instead of 'mixsd'.")
    if "outputlevel" in kwargs:
        stop("Use parameter 'output' instead of 'outputlevel'.")

    if scale == "estimate":
        use_ashr_grid = np.any(np.isin(("gridmult", "pointmass", "method"), list(kwargs)))
        if mode != "estimate" and not use_ashr_grid:
            scale = default_symmuni_scale(x, s, mode)[-1]
        else:
            scale = None

    if g_init is None:
        ash_res = ash(
            betahat=asvector(x),
            sebetahat=asvector(s),
            mode=mode,
            mixsd=scale,
            fixg=fix_g,
            outputlevel=ash_output(output),
            **kwargs,
        )

    else:
        if mode is not None or scale is not None:
            warnings.warn("mode and scale parameters are ignored when g_init is supplied.")

        ash_res = ash(
            betahat=asvector(x),
            sebetahat=asvector(s),
            g=g_init,
            fixg=fix_g,
            outputlevel=ash_output(output),
            **kwargs,
        )

    retlist = dict()

    if posterior_in_output(output):
        posterior = dict()

        if result_in_output(output):
            posterior["mean"] = ash_res["result"]["PosteriorMean"]
            posterior["sd"] = ash_res["result"]["PosteriorSD"]
            posterior["mean2"] = posterior["mean"] ** 2 + posterior["sd"] ** 2

        if lfsr_in_output(output):
            posterior["lfsr"] = ash_res["result"]["lfsr"]

        retlist = add_posterior_to_retlist(retlist, posterior, output)

    if g_in_output(output):
        retlist = add_g_to_retlist(retlist, ash_res["fitted_g"])

    if llik_in_output(output):
        retlist = add_llik_to_retlist(retlist, ash_res["loglik"])

    if sampler_in_output(output):
        retlist = add_sampler_to_retlist(retlist, ash_res["post_sampler"])

    return retlist
