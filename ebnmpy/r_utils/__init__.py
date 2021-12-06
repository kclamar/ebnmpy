import numpy as np


def stop(*args):
    raise Exception(*args)


def unlist(d: dict):
    return np.array(list(d.values()))


def numeric(*args, **kwargs):
    return np.zeros(*args)


def matrix(nrow, ncol):
    return np.empty((nrow, ncol))


def rep(x, times=1, each=1, length_out=None):
    if length_out is None:
        return np.tile(np.repeat(x, each), times)

    temp = rep(x, times, each, None)
    return np.tile(temp, int(np.ceil(length_out / len(temp))))[:length_out]


def pmax(a, b):
    return np.maximum(a, b)


def pmin(a, b):
    return np.minimum(a, b)


def length(x):
    if np.isscalar(x):
        return 1
    return len(x)


def asvector(x):
    return np.asarray(x)


def seq(a, b, by):
    return np.arange(a, b + by, by)
