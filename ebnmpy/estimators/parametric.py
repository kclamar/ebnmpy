from abc import abstractmethod

import numpy as np
from scipy.optimize import minimize

from ..opt_control_defaults import lbfgsb_control_defaults
from ..output import (
    add_g_to_retlist,
    add_llik_to_retlist,
    add_posterior_to_retlist,
    add_sampler_to_retlist,
    df_ret_str,
    g_in_output,
    g_ret_str,
    llik_in_output,
    llik_ret_str,
    posterior_in_output,
    samp_ret_str,
    sampler_in_output,
)
from ..workhorse_parametric import check_g_init, handle_optmethod_parameter
from .base import BaseEBNM


class ParametricEBNM(BaseEBNM):
    @property
    def _pointmass(self) -> bool:
        return True

    @property
    @abstractmethod
    def _class_name(self) -> str:
        pass

    @property
    def _scale_name(self) -> str:
        return "scale"

    @property
    def _mode_name(self) -> str:
        return "mean"

    def _fit(self, x, s, output, control):
        self._checkg(self.g_init, self.fix_g, self.mode, self.scale, self._pointmass)
        par_init = self._initpar(self.g_init, self.mode, self.scale, self._pointmass, x, s)

        if self.fix_g:
            fix_par = np.array([True, True, True])
        else:
            fix_par = np.array(
                [not self._pointmass, self.scale != "estimate", self.mode != "estimate"]
            )

        optmethod = handle_optmethod_parameter(self.optmethod, fix_par)

        x_optset = x
        s_optset = s

        if np.any(np.isinf(s)):
            x_optset = x[np.isfinite(s)]
            s_optset = s[np.isfinite(s)]

        optres = self._mle_parametric(
            x_optset,
            s_optset,
            par_init,
            fix_par,
            optmethod=optmethod["fn"],
            use_grad=optmethod["use_grad"],
            use_hess=optmethod["use_hess"],
            control=control,
        )

        retlist = dict()

        if posterior_in_output(output):
            posterior = self._summres(x, s, optres["par"], output)
            retlist = add_posterior_to_retlist(retlist, posterior, output)

        if g_in_output(output):
            fitted_g = self._partog(par=optres["par"])
            retlist = add_g_to_retlist(retlist, fitted_g)

        if llik_in_output(output):
            loglik = optres["val"]
            retlist = add_llik_to_retlist(retlist, loglik)

        if sampler_in_output(output):

            def post_sampler(nsamp):
                return self._postsamp(x, s, optres["par"], nsamp)

            retlist = add_sampler_to_retlist(retlist, post_sampler)

        if g_ret_str() in retlist:
            self.fitted_g_ = retlist[g_ret_str()]

        if df_ret_str() in retlist:
            self.posterior_ = retlist[df_ret_str()]

        if llik_ret_str() in retlist:
            self.log_likelihood_ = retlist[llik_ret_str()]

        if samp_ret_str() in retlist:
            self.posterior_sampler_ = retlist[samp_ret_str()]

    def _mle_parametric(self, x, s, par_init, fix_par, optmethod, use_grad, use_hess, control):
        scale_factor = 1 / np.median(s[s > 0])
        x = x * scale_factor
        s = s * scale_factor

        par_init = self._scalepar(par_init, scale_factor)
        precomp = self._precomp(x, s, par_init, fix_par)
        fn_params = dict(precomp, x=x, s=s, par_init=par_init, fix_par=fix_par)
        p = np.array(list(par_init.values()))[~np.array(fix_par)]

        if (not fix_par[1]) and np.isinf(p[0]):
            p[0] = np.sign(p[0]) * np.log(len(x))

        if all(fix_par):
            optpar = par_init
            optval = self._nllik(par=None, calc_grad=False, calc_hess=False, **fn_params)
        elif optmethod == "lbfgsb":
            control = dict(lbfgsb_control_defaults(), **control)

            def fn(par, kwargs):
                return self._nllik(par, calc_grad=False, calc_hess=False, **kwargs)

            if use_grad:

                def gr(par, kwargs):
                    return self._nllik(par, calc_grad=True, calc_hess=False, **kwargs)

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

        if isinstance(optpar, dict):
            retpar_values[~fix_par] = np.array(list(optpar.values()))[~fix_par]
        else:
            retpar_values[~fix_par] = optpar

        retpar = dict(zip(list(retpar), retpar_values))

        retpar = self._scalepar(par=retpar, scale_factor=1 / scale_factor)
        optval = optval - sum(np.isfinite(x) * np.log(scale_factor))

        retlist = self._postcomp(
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

    def _checkg(self, g_init, fix_g, mode, scale, pointmass):
        return check_g_init(
            g_init=g_init,
            fix_g=fix_g,
            mode=mode,
            scale=scale,
            pointmass=pointmass,
            class_name=self._class_name,
            scale_name=self._scale_name,
            mode_name=self._mode_name,
        )
