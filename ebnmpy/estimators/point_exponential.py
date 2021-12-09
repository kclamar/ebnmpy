from ..point_exponential import (
    pe_initpar,
    pe_nllik,
    pe_partog,
    pe_postcomp,
    pe_postsamp,
    pe_precomp,
    pe_scalepar,
    pe_summres,
)
from .parametric import ParametricEBNM


class PointExponentialEBNM(ParametricEBNM):
    @property
    def _class_name(self) -> str:
        return "gammamix"

    @property
    def _mode_name(self) -> str:
        return "shift"

    def _initpar(self, g_init, mode, scale, pointmass, x, s):
        return pe_initpar(g_init, mode, scale, pointmass, x, s)

    def _scalepar(self, par, scale_factor):
        return pe_scalepar(par, scale_factor)

    def _precomp(self, x, s, par_init, fix_par):
        return pe_precomp(x, s, par_init, fix_par)

    def _nllik(self, par, x, s, par_init, fix_par, calc_grad, calc_hess, **kwargs):
        return pe_nllik(par, x, s, par_init, fix_par, calc_grad, calc_hess)

    def _postcomp(self, optpar, optval, x, s, par_init, fix_par, scale_factor, **kwargs):
        return pe_postcomp(optpar, optval, x, s, par_init, fix_par, scale_factor)

    def _summres(self, x, s, optpar, output):
        return pe_summres(x, s, optpar, output)

    def _partog(self, par):
        return pe_partog(par)

    def _postsamp(self, x, s, optpar, nsamp):
        return pe_postsamp(x, s, optpar, nsamp)
