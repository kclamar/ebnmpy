from .ebnm import ebnm_workhorse
from .output import output_default


def ebnm_point_normal(
    x,
    s=1,
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
        prior_family="point_normal",
    )


def ebnm_point_laplace(
    x,
    s=1,
    mode=0,
    scale="estimate",
    g_init=None,
    fix_g=False,
    output=output_default(),
    optmethod="nograd_lbfgsb",
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
        prior_family="point_laplace",
    )


def ebnm_point_exponential(
    x,
    s=1,
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
        prior_family="point_exponential",
    )


def ebnm_normal(
    x,
    s=1,
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
        prior_family="normal",
    )


def ebnm_horseshoe(
    x, s=1, scale="estimate", g_init=None, fix_g=False, output=output_default(), control=None
):
    return ebnm_workhorse(
        x=x,
        s=s,
        mode=0,
        scale=scale,
        g_init=g_init,
        fix_g=fix_g,
        output=output,
        optmethod=None,
        control=control,
        prior_family="horseshoe",
    )


def ebnm_normal_scale_mixture(
    x,
    s=1,
    mode=0,
    scale="estimate",
    g_init=None,
    fix_g=False,
    output=output_default(),
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
        optmethod=None,
        control=control,
        prior_family="normal_scale_mixture",
    )


def ebnm_unimodal(
    x,
    s=1,
    mode=0,
    scale="estimate",
    g_init=None,
    fix_g=False,
    output=output_default(),
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
        optmethod=None,
        control=control,
        prior_family="unimodal",
    )


def ebnm_unimodal_symmetric(
    x,
    s=1,
    mode=0,
    scale="estimate",
    g_init=None,
    fix_g=False,
    output=output_default(),
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
        optmethod=None,
        control=control,
        prior_family="unimodal_symmetric",
    )


def ebnm_unimodal_nonnegative(
    x,
    s=1,
    mode=0,
    scale="estimate",
    g_init=None,
    fix_g=False,
    output=output_default(),
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
        optmethod=None,
        control=control,
        prior_family="unimodal_nonnegative",
    )


def ebnm_unimodal_nonpositive(
    x,
    s=1,
    mode=0,
    scale="estimate",
    g_init=None,
    fix_g=False,
    output=output_default(),
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
        optmethod=None,
        control=control,
        prior_family="unimodal_nonpositive",
    )


def ebnm_ash(
    x,
    s=1,
    mode=0,
    scale="estimate",
    g_init=None,
    fix_g=False,
    output=output_default(),
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
        optmethod=None,
        control=control,
        prior_family="ash",
    )


def ebnm_npmle(
    x,
    s=1,
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
        mode=0,
        scale=scale,
        g_init=g_init,
        fix_g=fix_g,
        output=output,
        optmethod=optmethod,
        control=control,
        prior_family="npmle",
    )


def ebnm_deconvolver(
    x,
    s=1,
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
        mode=0,
        scale=scale,
        g_init=g_init,
        fix_g=fix_g,
        output=output,
        optmethod=optmethod,
        control=control,
        prior_family="deconvolver",
    )
