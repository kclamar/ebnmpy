import warnings
from abc import abstractmethod
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y

from ..ebnm import check_args, handle_mode_parameter, handle_scale_parameter
from ..output import lfsr_arg_str, output_default


class BaseEBNM(BaseEstimator):
    n_features_in_: int
    s_: np.ndarray
    posterior_: dict
    fitted_g_: dict
    log_likelihood_: float
    posterior_sampler_: Callable

    def __init__(
        self,
        s=1,
        mode=0,
        scale="estimate",
        g_init=None,
        fix_g=False,
        output=output_default(),
        optmethod=None,
        control=None,
        include_posterior_sampler=False,
    ):
        self.s = s
        self.mode = mode
        self.scale = scale
        self.g_init = g_init
        self.fix_g = fix_g
        self.output = output
        self.optmethod = optmethod
        self.control = control
        self.include_posterior_sampler = include_posterior_sampler

    def _fit(self, x, s, output, control):
        pass

    def fit(self, x, y=None, s=None):
        x, s, output, control = self._check_args(x, y, s)
        self._fit(x, s, output, control)
        self.n_features_in_ = 1 if len(x.shape) == 1 else x.shape[-1]

        return self

    def _check_args(self, x, y, s):
        if isinstance(x, (np.ndarray, np.memmap, tuple, list)):
            x = np.array(x)

        if s is None:
            s = self.s

        if y is not None:
            x, y = check_X_y(x, y)

        if np.isscalar(s):
            s = np.full_like(x, s)
        else:
            s = np.asarray(s).copy()

        check_args(x, s, self.g_init, self.fix_g, self.output, self.mode)

        mode = handle_mode_parameter(self.mode)
        handle_scale_parameter(self.scale)

        if lfsr_arg_str() in self.output and mode != 0:
            warnings.warn(
                "Since they're not well defined for nonzero modes, local false sign rates won't be "
                "returned. "
            )
            output = tuple(i for i in self.output if i != lfsr_arg_str())
        else:
            output = self.output

        control = dict() if self.control is None else self.control

        return x, s, output, control

    @abstractmethod
    def _checkg(self, g_init, fix_g, mode, scale, pointmass):
        pass

    @abstractmethod
    def _initpar(self, g_init, mode, scale, pointmass, x, s):
        pass

    @abstractmethod
    def _scalepar(self, par, scale_factor):
        pass

    @abstractmethod
    def _precomp(self, x, s, par_init, fix_par):
        pass

    @abstractmethod
    def _nllik(self, par, x, s, par_init, fix_par, calc_grad, calc_hess, **kwargs):
        pass

    @abstractmethod
    def _postcomp(self, optpar, optval, x, s, par_init, fix_par, scale_factor, **kwargs):
        pass

    @abstractmethod
    def _summres(self, x, s, optpar, output):
        pass

    @abstractmethod
    def _partog(self, par):
        pass

    @abstractmethod
    def _postsamp(self, x, s, optpar, nsamp):
        pass

    def sample(self, n):
        check_is_fitted(self)

        if hasattr(self, "posterior_sampler_"):
            return self.posterior_sampler_(n)

        raise Exception(
            f"To sample from the posterior, pass include_posterior_sampler=True when "
            f"initializing {type(self).__name__}."
        )
