import os
import pkgutil
import io
import numpy as np


def target_timeseries_pred(alpha_now, beta_now, tau2_now, proxy_ts, prior_pars):
    """

    Parameters
    ----------
    alpha_now : scalar
    beta_now : scalar
    tau2_now : scalar
        Current value of the residual variance.
    proxy_ts : ndarray
        Time series of the proxy. Assume no temporal structure for now, as
        timing is not equal.
    prior_pars : dict
        mu : ndarray
            Prior means for each element of the time series.
        inv_cov: ndarray
            Inverse of the prior covariance matrix for the time series.

    Returns
    -------
    Sample of target time series vector conditional on the rest.
    """
    # TODO(brews): Above docstring is based on original MATLAB. Needs cleanup.
    n_ts = len(proxy_ts)

    # Inverse posterior covariance matrix
    inv_post_cov = prior_pars['inv_cov'] + beta_now ** 2 / tau2_now * np.eye(n_ts)

    # Used cholesky to speed things up!
    post_cov = np.linalg.solve(inv_post_cov, np.eye(n_ts))
    sqrt_post_cov = np.linalg.cholesky(post_cov).T
    # Get first factor for the mean
    mean_first_factor = prior_pars['inv_cov'] @ prior_pars['mu'] + (1/tau2_now) * beta_now * (proxy_ts - alpha_now)
    mean_full = post_cov @ mean_first_factor

    timeseries_pred = mean_full + sqrt_post_cov @ np.random.randn(n_ts).T

    return timeseries_pred


def get_example_data(filename):
    """Get a BytesIO object for a bayspar example file.

    Parameters
    ----------
    filename : str
        File to load.

    Returns
    -------
    BytesIO of the example file.
    """
    resource_str = os.path.join('example_data', filename)
    return io.BytesIO(pkgutil.get_data('bayspar', resource_str))
