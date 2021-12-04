import numpy as np


def rnorm(n, mean=0, sd=1):
    return np.random.normal(loc=mean, scale=sd, size=n)
