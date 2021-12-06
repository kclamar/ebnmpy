import numpy as np
from scipy.stats import expon, norm, truncnorm


def dnorm(x, mean=0, sd=1, log=False):
    g = norm(mean, sd)
    if log:
        return g.logpdf(x)
    return g.pdf(x)


def pnorm(q, mean=0, sd=1, lower_tail=True, log_p=False):
    if not lower_tail:
        q = mean * 2 - q
    g = norm(mean, sd)
    if log_p:
        return g.logcdf(q)
    return g.cdf(q)


def rnorm(n, mean=0, sd=1):
    return np.random.normal(loc=mean, scale=sd, size=n)


def rbinom(n, size, prob):
    return np.random.binomial(n=size, p=prob, size=n)


def rtruncnorm(n, a=-np.inf, b=np.inf, mean=0, sd=1):
    return truncnorm.rvs(a=a, b=b, loc=mean, scale=sd, size=n)


def rexp(n, rate):
    return expon.rvs(loc=0, scale=1 / rate, size=n)
