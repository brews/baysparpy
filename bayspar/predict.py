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
    output : Prediction

    Raises
    ------
    EnsembleSizeError
    """
    draws = get_draws(temptype)

    nd = len(seatemp)
    ntk = draws.alpha_samples_comp.shape[1]
    if ntk < nens:
        raise EnsembleSizeError(ntk, nens)

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
    output : Prediction

    Raises
    ------
    EnsembleSizeError
    """
    draws = get_draws(temptype)
    obs = get_seatemp(temptype)

    nd = len(tex)
    ntk = draws.alpha_samples_comp.shape[1]
    if ntk < nens:
        raise EnsembleSizeError(ntk, nens)

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
    progressbar: bool
        Whether or not to display a progress bar on the command line. The bar
        shows how many analogs have been completed.

    Returns
    -------
    output : Prediction

    Raises
    ------
    EnsembleSizeError
    """
    draws = get_draws(temptype)
    tex_obs = get_tex(temptype)

    nd = len(tex)
    ntk = draws.alpha_samples_comp.shape[1]
    if ntk < nens:
        raise EnsembleSizeError(ntk, nens)

    latlon_match, val_match = tex_obs.find_within_tolerance(x=tex.mean(),
                                                            tolerance=search_tol)
    n_locs_g = len(latlon_match)

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


class EnsembleSizeError(Exception):
    """Raised when user requests too large an ensemble for prediction

    Parameters
    ----------
    available_size : int
        The available ensemble size.
    requested_size : int
        The user-requested ensemble size.
    """
    def __init__(self, available_size, requested_size):
        self.available_size = available_size
        self.requested_size = requested_size
