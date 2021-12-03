from .ebnm import ebnm_workhorse
from .output import output_default


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
