def nlm_control_defaults():
    return dict(ndigit=8, stepmax=5, check_analyticals=False)


def lbfgsb_control_defaults():
    return dict()


def trust_control_defaults():
    return dict(rinit=2, rmax=5)


def optimize_control_defaults():
    return dict()
