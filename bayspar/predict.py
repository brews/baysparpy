import numpy as np

from bayspar.utils import target_timeseries_pred
from bayspar.posterior import get_draws
from bayspar.observations import get_seatemp, get_tex


def predict_tex(dats, lat, lon, temptype, nens=5000, save_ensemble=False):
    """Predict TEX86 from sea temperature

    Parameters
    ----------
    dats : ndarray
        n-length array of sea temperature observations (°C) from a single
        location.
    lat : float
        Site latitude from -90 to 90.
    lon : float
        Site longitude from -180 to 180.
    temptype : str
        Type of sea temperature used. Either 'sst' for sea-surface or 'subt'.
    nens : int
        Size of MCMC ensemble draws to use for calculation.
    save_ensemble : bool
        Should the entire MCMC ensemble be returned? If not then just
        percentiles.

    Returns
    -------
    output : dict
        preds : ndarray
            Predicted TEX86 percentiles.
        siteloc : tuple
            Site (lat, lon).
        gridloc': sequence
            Latlon of the observation gridpoint used as prior mean.
        predsens': ndarray or None
            If 'save_ensemble' is True, this is the full MCMC ensemble for TEX86
            prediction. Otherwise, is None.
    """
    # TODO(brews): Write predict_tex() function.
    assert temptype in ['sst', 'subt']
    assert -180 <= lon <= 180
    assert -90 <= lat <= 90

    nd = len(dats)
    pers3 = (np.round(np.array([0.05, 0.50, 0.95]) * nens) - 1).astype(int)

    draws = get_draws(temptype)

    ntk = draws.alpha_samples_comp.shape[1]
    assert ntk > nens

    alpha_samples_comp, beta_samples_comp = draws.find_alphabeta_near(lat=lat,
                                                                      lon=lon)
    tau2_samples = draws.tau2_samples

    grid_latlon = draws.find_nearest_latlon(lat=lat, lon=lon)

    tex = np.empty((nd, nens))
    for i in range(nens):
        tau2_now = tau2_samples[i]
        beta_now = beta_samples_comp[i]
        alpha_now = alpha_samples_comp[i]
        tex[:, i] = np.random.normal(dats * beta_now + alpha_now,
                                     np.sqrt(tau2_now))

    tex_s = np.sort(tex, axis=1)

    output = {'preds': tex_s[:, pers3],
              'siteloc': (lat, lon),
              'gridloc': tuple(grid_latlon),
              'predsens': None}

    if save_ensemble:
        output['predsens'] = tex

    return output


def predict_sst(*args, **kwargs):
    """Predict sea surface temperature with TEX86
    """
    return predict_seatemp(*args, temptype='sst', **kwargs)


def predict_subt(*args, **kwargs):
    """Predict sub-surface sea temperature with TEX86
    """
    return predict_seatemp(*args, temptype='subt', **kwargs)


def predict_seatemp(dats, lat, lon, prior_std, temptype, prior_mean=None, nens=5000,
                    save_ensemble=False):
    """Predict sea temperature with TEX86

    Parameters
    ----------
    dats : ndarray
        n-length array of TEX86 observations from a single location.
    lat : float
        Site latitude from -90 to 90.
    lon : float
        Site longitude from -180 to 180.
    prior_std : float
        Prior standard deviation for sea temperature (°C).
    temptype : str
        Type of sea temperature desired. Either 'sst' for sea-surface or 'subt'.
    prior_mean : float or None, optional
        Prior mean for sea temperature (°C). If 'None', the prior mean is found
        by searching for a "close" value in observed sea temperature records.
    nens : int
        Size of MCMC ensemble draws to use for calculation.
    save_ensemble : bool
        Should the entire MCMC ensemble be returned? If not, then just
        percentiles.

    Returns
    -------
    output : dict
        preds : ndarray
            Predicted sea temperature percentiles.
        siteloc : tuple
            Site (lat, lon).
        gridloc': sequence
            Latlon of the observation gridpoint used as prior mean.
        priormean': float
            Sea temperature (°C) prior mean from observation.
        priorstd': float
            Sea temperature (°C) prior standard deviation.
        predsens': ndarray or None
            If 'save_ensemble' is True, this is the full MCMC ensemble for sea
            temperature prediction. Otherwise, is None.
    """
    assert temptype in ['sst', 'subt']
    assert -180 <= lon <= 180
    assert -90 <= lat <= 90

    draws = get_draws(temptype)
    obs = get_seatemp(temptype)

    # TODO(brews): trim tau**2 (may not have burnin) ln 92 of bayspar_tex.m
    ntk = draws.alpha_samples_comp.shape[1]
    assert ntk > nens
    # TODO(brews): trim posterior draws to "sample full span of ensemble" (ln 88-101 of bayspar_tex.m)

    nd = len(dats)

    if prior_mean is None:
        close_obs, close_dist = obs.get_close_obs(lat=lat, lon=lon)
        prior_mean = close_obs.mean()

    alpha_samples_comp, beta_samples_comp = draws.find_alphabeta_near(lat=lat, lon=lon)
    tau2_samples = draws.tau2_samples

    grid_latlon = draws.find_nearest_latlon(lat=lat, lon=lon)

    prior_par = {'mu': np.ones(nd) * prior_mean,
                 'inv_cov': np.eye(nd) * prior_std ** -2}

    preds = np.empty((nd, nens))
    for jj in range(nens):
        preds[:, jj] = target_timeseries_pred(alpha_now=alpha_samples_comp[jj],
                                              beta_now=beta_samples_comp[jj],
                                              tau2_now=tau2_samples[jj],
                                              proxy_ts=dats,
                                              prior_pars=prior_par)
        # TODO(brews): Consider a progress bar for this loop.

    preds_s = np.sort(preds, axis=1)

    pers3 = (np.round(np.array([0.05, 0.50, 0.95]) * nens) - 1).astype(int)
    output = {'preds': preds_s[:, pers3],
              'siteloc': (lat, lon),
              'gridloc': tuple(grid_latlon),
              'priormean': prior_mean,
              'priorstd': prior_std,
              'predsens': None}

    if save_ensemble:
        output['predsens'] = preds

    return output


def predict_seatemp_analog(dats, prior_std, temptype, search_tol,
                           prior_mean=None, nens=5000, save_ensemble=False):
    """Predict sea temperature with TEX86, using the analog method

    Parameters
    ----------
    dats : ndarray
        n-length array of TEX86 observations from a single location.
    prior_std : float
        Prior standard deviation for sea temperature (°C).
    temptype : str
        Type of sea temperature desired. Either 'sst' for sea-surface or 'subt'.
    search_tol: float
        Tolerance for finding analog locations. Comparison is between the mean
        of dats and the mean tex value within each large gridcell.
    prior_mean : float
        Prior mean for sea temperature (°C).
    nens : int
        Size of MCMC ensemble draws to use for calculation.
    save_ensemble : bool
        Should the entire MCMC ensemble be returned? If not, then just
        percentiles.

    Returns
    -------
    output : dict
        preds : ndarray
            Predicted sea temperature percentiles.
        anlocs : sequence
            Sequence of (lat, lon) used for the analog.
        priormean: float
            Sea temperature (°C) prior mean.
        priorstd: float
            Sea temperature (°C) prior standard deviation.
        predsens': ndarray or None
            If 'save_ensemble' is True, this is the full MCMC ensemble for sea
            temperature prediction. Otherwise, is None.
    """
    assert temptype in ['sst', 'subt']

    draws = get_draws(temptype)
    tex_obs = get_tex(temptype)

    # TODO(brews): trim tau**2 (may not have burnin) ln 92 of bayspar_tex.m
    ntk = draws.alpha_samples_comp.shape[1]
    assert ntk > nens
    # TODO(brews): trim posterior draws to "sample full span of ensemble" (ln 88-101 of bayspar_tex.m)

    nd = len(dats)

    latlon_match, val_match = tex_obs.find_within_tolerance(x=dats.mean(),
                                                            tolerance=search_tol)
    n_locs_g = len(latlon_match)
    assert n_locs_g > 0

    prior_par = {'mu': np.ones(nd) * prior_mean,
                 'inv_cov': np.eye(nd) * prior_std ** -2}

    preds = np.empty((nd, n_locs_g, nens))
    for kk, latlon in enumerate(latlon_match):
        alpha_samples, beta_samples = draws.find_alphabeta_near(*latlon)
        for jj in range(nens):
            a_now = alpha_samples[jj]
            b_now = beta_samples[jj]
            t2_now = draws.tau2_samples[jj]
            preds[:, kk, jj] = target_timeseries_pred(alpha_now=a_now,
                                                      beta_now=b_now,
                                                      tau2_now=t2_now,
                                                      proxy_ts=dats,
                                                      prior_pars=prior_par)
            # TODO(brews): Consider a progress bar for this loop.

    pers3 = (np.round(np.array([0.05, 0.50, 0.95]) * nens * n_locs_g) - 1).astype(int)
    preds_stack = np.reshape(preds, (nd, nens * n_locs_g), order='F')
    preds_s = np.sort(preds_stack, axis=1)

    output = {'preds': preds_s[:, pers3],
              'anlocs': latlon_match,
              'priormean': prior_mean,
              'priorstd': prior_std,
              'predsens': None}

    if save_ensemble:
        output['predsens'] = preds

    return output
