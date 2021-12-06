from .truncnorm import my_e2truncnorm, my_etruncnorm


def normalmix(pi, mean, sd):
    return dict(pi=pi, mean=mean, sd=sd)
