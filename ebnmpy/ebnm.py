import warnings

from .output import as_ebnm, lfsr_arg_str
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
from .workhorse_parametric import parametric_workhorse


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
        output.pop(lfsr_arg_str())

    if prior_family == "normal":
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
    else:
        raise NotImplementedError

    return as_ebnm(retlist)


def check_args(x, s, g_init, fix_g, output, mode):
    # TODO
    pass


def handle_mode_parameter(mode):
    if mode == "estimate":
        pass
    else:
        # TODO
        raise NotImplementedError
    return mode


def handle_scale_parameter(scale):
    if scale == "estimate":
        pass
    else:
        # TODO
        raise NotImplementedError
    return scale
