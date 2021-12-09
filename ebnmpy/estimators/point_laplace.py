from ..point_laplace import (
    pl_initpar,
    pl_nllik,
    pl_partog,
    pl_postcomp,
    pl_postsamp,
    pl_precomp,
    pl_scalepar,
    pl_summres,
)
from .parametric import ParametricEBNM


class PointLaplaceEBNM(ParametricEBNM):
    @property
    def _class_name(self) -> str:
        return "laplacemix"

    def _initpar(self, g_init, mode, scale, pointmass, x, s):
        return pl_initpar(g_init, mode, scale, pointmass, x, s)

    def _scalepar(self, par, scale_factor):
        return pl_scalepar(par, scale_factor)

    def _precomp(self, x, s, par_init, fix_par):
        return pl_precomp(x, s, par_init, fix_par)

    def _nllik(self, par, x, s, par_init, fix_par, calc_grad, calc_hess, **kwargs):
        return pl_nllik(par, x, s, par_init, fix_par, calc_grad, calc_hess)

    def _postcomp(self, optpar, optval, x, s, par_init, fix_par, scale_factor, **kwargs):
        return pl_postcomp(optpar, optval, x, s, par_init, fix_par, scale_factor)

    def _summres(self, x, s, optpar, output):
        return pl_summres(x, s, optpar, output)

    def _partog(self, par):
        return pl_partog(par)

    def _postsamp(self, x, s, optpar, nsamp):
        return pl_postsamp(x, s, optpar, nsamp)
