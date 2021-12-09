from ..point_normal import (
    pn_initpar,
    pn_nllik,
    pn_partog,
    pn_postcomp,
    pn_postsamp,
    pn_precomp,
    pn_scalepar,
    pn_summres,
)
from .parametric import ParametricEBNM


class PointNormalEBNM(ParametricEBNM):
    @property
    def _class_name(self) -> str:
        return "normalmix"

    @property
    def _scale_name(self) -> str:
        return "sd"

    def _initpar(self, g_init, mode, scale, pointmass, x, s):
        return pn_initpar(g_init, mode, scale, pointmass, x, s)

    def _scalepar(self, par, scale_factor):
        return pn_scalepar(par, scale_factor)

    def _precomp(self, x, s, par_init, fix_par):
        return pn_precomp(x, s, par_init, fix_par)

    def _nllik(self, par, x, s, par_init, fix_par, calc_grad, calc_hess, **kwargs):
        return pn_nllik(
            par=par,
            x=x,
            s=s,
            par_init=par_init,
            fix_par=fix_par,
            calc_grad=calc_grad,
            calc_hess=calc_hess,
            **kwargs,
        )

    def _postcomp(self, optpar, optval, x, s, par_init, fix_par, scale_factor, **kwargs):
        return pn_postcomp(
            optpar=optpar,
            optval=optval,
            x=x,
            s=s,
            par_init=par_init,
            fix_par=fix_par,
            scale_factor=scale_factor,
            **kwargs,
        )

    def _summres(self, x, s, optpar, output):
        return pn_summres(x, s, optpar, output)

    def _partog(self, par):
        return pn_partog(par)

    def _postsamp(self, x, s, optpar, nsamp):
        return pn_postsamp(x, s, optpar, nsamp)
