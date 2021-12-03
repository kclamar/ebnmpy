def output_default():
    return pm_arg_str(), psd_arg_str(), g_arg_str(), llik_arg_str()


def output_all():
    return (
        pm_arg_str(),
        psd_arg_str(),
        pm2_arg_str(),
        lfsr_arg_str(),
        g_arg_str(),
        llik_arg_str(),
        samp_arg_str(),
    )


def pm_arg_str():
    return "posterior_mean"


def psd_arg_str():
    return "posterior_sd"


def pm2_arg_str():
    return "posterior_second_moment"


def lfsr_arg_str():
    return "lfsr"


def g_arg_str():
    return "fitted_g"


def llik_arg_str():
    return "log_likelihood"


def samp_arg_str():
    return "posterior_sampler"


def df_ret_str():
    return "posterior"


def pm_ret_str():
    return "mean"


def psd_ret_str():
    return "sd"


def pm2_ret_str():
    return "second_moment"


def lfsr_ret_str():
    return "lfsr"


def g_ret_str():
    return "fitted_g"


def llik_ret_str():
    return "log_likelihood"


def samp_ret_str():
    return "posterior_sampler"


def as_ebnm(retlist):
    return retlist


def lfsr_in_output(output):
    return lfsr_arg_str() in output


def result_in_output(output):
    res_args = pm_arg_str(), psd_arg_str(), pm2_arg_str()
    return any(i in output for i in res_args)


def posterior_in_output(output):
    post_args = pm_arg_str(), psd_arg_str(), pm2_arg_str(), lfsr_arg_str()
    return any(i in output for i in post_args)


def add_posterior_to_retlist(retlist, posterior, output):
    df = dict()
    if pm_arg_str() in output:
        df[pm_ret_str()] = posterior["mean"]
    if psd_arg_str() in output:
        df[psd_ret_str()] = posterior["sd"]
    if pm2_arg_str() in output:
        df[pm2_ret_str()] = posterior["mean2"]
    if lfsr_arg_str() in output:
        df[lfsr_ret_str()] = posterior["lfsr"]

    retlist[df_ret_str()] = df
    return retlist


def g_in_output(output):
    return g_arg_str() in output


def add_g_to_retlist(retlist, g):
    retlist[g_ret_str()] = g
    return retlist


def llik_in_output(output):
    return llik_arg_str() in output


def add_llik_to_retlist(retlist, llik):
    retlist[llik_ret_str()] = llik
    return retlist


def sampler_in_output(output):
    return samp_arg_str() in output


def add_sampler_to_retlist(retlist, sampler):
    retlist[samp_ret_str()] = sampler
    return retlist
