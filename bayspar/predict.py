import numpy as np
import attr
import attr.validators as av
from tqdm import tqdm

from bayspar.utils import target_timeseries_pred
from bayspar.modelparams import get_draws
from bayspar.observations import get_seatemp, get_tex


@attr.s()
class Prediction:
    """MCMC prediction

    Parameters
    ----------
    ensemble : ndarray
        Ensemble of predictions. A 2d array (nxm) for n predictands and m
        ensemble members.
    temptype : str
        Type of sea temperature used for prediction.
    latlon : tuple or None, optional
        Optional tuple of the site location (lat, lon).
    prior_mean : float or None, optional
        Prior mean used for the prediction.
    prior_std : float or None, optional
        Prior sample standard deviation used for the prediction.
    modelparam_gridpoints : list of tuples or None, optional
        A list of one or more (lat, lon) points used to collect
        spatially-sensitive model parameters.
    analog_gridpoints : list of tuples or None, optional
        A list of one or more (lat, lon) points used for an analog prediction.
    """
    temptype = attr.ib()
    ensemble = attr.ib(validator=av.optional(av.instance_of(np.ndarray)))
    latlon = attr.ib(default=None,
                     validator=av.optional(av.instance_of(tuple)))
    prior_mean = attr.ib(default=None)
    prior_std = attr.ib(default=None)
    modelparam_gridpoints = attr.ib(default=None,
                                    validator=av.optional(av.instance_of(list)))
    analog_gridpoints = attr.ib(default=None,
                                validator=av.optional(av.instance_of(list)))

    def percentile(self, q=None, interpolation='nearest'):
        """Compute the qth ranked percentile from ensemble members.

        Parameters
        ----------
        q : float ,sequence of floats, or None, optional
            Percentiles (i.e. [0, 100]) to compute. Default is 5%, 50%, 95%.
        interpolation : str, optional
            Passed to numpy.percentile. Default is 'nearest'.

        Returns
        -------
        perc : ndarray
            A 2d (nxm) array of floats where n is the number of predictands in
            the ensemble and m is the number of percentiles ('len(q)').
        """
        if q is None:
            q = [5, 50, 95]
        q = np.array(q, dtype=np.float64, copy=True)

        # Because analog ensembles have 3 dims
        target_axis = list(range(self.ensemble.ndim))[1:]

        perc = np.percentile(self.ensemble, q=q, axis=target_axis,
                             interpolation=interpolation)
        return perc.T


def predict_tex(seatemp, lat, lon, temptype, nens=5000):
    """Predict TEX86 from sea temperature

    Parameters
    ----------
    seatemp : ndarray
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

    nd = len(seatemp)

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
        tex[:, i] = np.random.normal(seatemp * beta_now + alpha_now,
                                     np.sqrt(tau2_now))

    output = Prediction(ensemble=tex,
                        temptype=temptype,
                        latlon=(lat, lon),
                        modelparam_gridpoints=[tuple(grid_latlon)])
    return output


def predict_seatemp(tex, lat, lon, prior_std, temptype, prior_mean=None, nens=5000):
    """Predict sea temperature with TEX86

    Parameters
    ----------
    tex : ndarray
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
    # TODO(brews): trim modelparams draws to "sample full span of ensemble" (ln 88-101 of bayspar_tex.m)

    nd = len(tex)

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
                                              proxy_ts=tex,
                                              prior_pars=prior_par)
        # TODO(brews): Consider a progress bar for this loop.

    output = Prediction(ensemble=preds,
                        temptype=temptype,
                        latlon=(lat, lon),
                        prior_mean=prior_mean,
                        prior_std=prior_std,
                        modelparam_gridpoints=[tuple(grid_latlon)])
    return output


def predict_seatemp_analog(tex, prior_std, temptype, search_tol, prior_mean=None, nens=5000, progressbar=True):
    """Predict sea temperature with TEX86, using the analog method

    Parameters
    ----------
    tex : ndarray
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
    progressbar: bool
        Whether or not to display a progress bar on the command line. The bar
        shows how many analogs have been completed.

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
    # TODO(brews): trim modelparams draws to "sample full span of ensemble" (ln 88-101 of bayspar_tex.m)

    nd = len(tex)

    latlon_match, val_match = tex_obs.find_within_tolerance(x=tex.mean(),
                                                            tolerance=search_tol)
    n_locs_g = len(latlon_match)
    assert n_locs_g > 0

    prior_par = {'mu': np.ones(nd) * prior_mean,
                 'inv_cov': np.eye(nd) * prior_std ** -2}

    n_latlon_matches = len(latlon_match)
    indices = range(n_latlon_matches)

    if progressbar:
        indices = tqdm(indices, total=n_latlon_matches)

    preds = np.empty((nd, n_locs_g, nens))
    for kk in indices:
        latlon = latlon_match[kk]
        alpha_samples, beta_samples = draws.find_alphabeta_near(*latlon)
        for jj in range(nens):
            a_now = alpha_samples[jj]
            b_now = beta_samples[jj]
            t2_now = draws.tau2_samples[jj]
            preds[:, kk, jj] = target_timeseries_pred(alpha_now=a_now,
                                                      beta_now=b_now,
                                                      tau2_now=t2_now,
                                                      proxy_ts=tex,
                                                      prior_pars=prior_par)

    output = Prediction(ensemble=preds,
                        temptype=temptype,
                        prior_mean=prior_mean,
                        prior_std=prior_std,
                        analog_gridpoints=latlon_match)
    return output
