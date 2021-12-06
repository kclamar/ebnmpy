from .r_utils import length


def default_symmuni_scale(x, s, mode=0, min_K=3, max_K=300, KLdiv_target=None):
    if KLdiv_target is None:
        KLdiv_target = 1 / length(x)

    # TODO
    raise NotImplementedError
