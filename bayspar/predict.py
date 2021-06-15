"""
    baysparpy
    =========

    An Open Source Python package for TEX86 calibration.

    This package is based on the original BAYSPAR 
    (BAYesian SPAtially-varying Regression) for MATLAB 
    (https://github.com/jesstierney/BAYSPAR).

    Originator: Steven Brewster Malevich
                University of Arizona Department of Geosciences

    Revisions:  Mingsong Li
                Penn State Geosciences
    Date:       Sept 23, 2019

    Revision: Mingsong Li
                Peking University
    Date:       Jun 16, 2021
                     
    Purpose: add predict_tex_analog module
             
            : TEX_forward model for analog model of baysparpy
            to simplify the code and save space
            
            add get_draws_analog module
            : Get Draws instance for a draw type
            for the analog model
    
    New files added: alpha_samples.mat
                     beta_samples.mat
            These two files were trimmed as they may not have had burnin removed
            They were calculated using the following matlab code 
            in TEX_forward.m of BAYSPAR
            https://github.com/jesstierney/BAYSPAR/blob/master/TEX_forward.m

        Ntk = 20000;
        load('alpha_samples.mat', 'alpha_samples')
        alpha_samples=[alpha_samples.field];
        alpha_samples=alpha_samples(:,end-Ntk+1:end);
        load('beta_samples.mat')
        beta_samples=[beta_samples.field];
        beta_samples=beta_samples(:,end-Ntk+1:end);
        save('alpha_samples','alpha_samples');
        save('beta_samples','beta_samples');
"""
import numpy as np
import numpy.matlib
import attr
import attr.validators as av
from tqdm import tqdm
import sys

from bayspar.utils import target_timeseries_pred
from bayspar.modelparams import get_draws, get_draws_analog
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

def predict_tex_analog(seatemp, temptype = 'sst', search_tol = 5., nens=5000):
    """Predict TEX86 from sea temperature analog model

    Parameters
    ----------
    seatemp : ndarray
            n-length array of sea temperature observations (°C) from a single location.
    temptype : str 
            Type of sea temperature used. Either 'sst' (default) for sea-surface or 'subt'.
    search_tol: float 
            search tolerance in seatemp units (required for analog mode)
    nens : int 
            Size of MCMC ensemble draws to use for calculation.
        
    Returns
    -------
    output : Prediction
    Raises
    ------
    EnsembleSizeError
    """
    draws = get_draws_analog(temptype)
    tex_obs = get_tex(temptype)
    
    nd = len(seatemp)  # number of input sea temperature data
    
    ntk = draws.alpha_samples_comp.shape[1]
    if ntk < nens:
        raise EnsembleSizeError(ntk, nens)
        
    latlon_match, val_match, inder_g = tex_obs.find_t_within_tolerance(t=seatemp.mean(),
                                                                       tolerance=search_tol)
    
    if inder_g.size == 0:
        sys.exit('No analogs were found. Check seatemp or make your search tolerance wider.')
        
    alpha_samples1 = draws.alpha_samples_comp[inder_g]
    
    beta_samples1 = draws.beta_samples_comp[inder_g]
    tau2_samples1 = draws.tau2_samples.reshape((draws.tau2_samples.size,1)).T

    tau2_sample_reshapen = int(alpha_samples1.shape[0] * alpha_samples1.shape[1] / draws.tau2_samples.size)
    
    alpha_samples = np.reshape(alpha_samples1, (alpha_samples1.size, ))
    beta_samples = np.reshape(beta_samples1, (beta_samples1.size, ))
    
    tau2_samples2 = np.matlib.repmat(tau2_samples1, 1, tau2_sample_reshapen)
    tau2_samples = np.reshape(tau2_samples2, (tau2_samples2.size, ))
    
    # downsample to nens = 5000
    iters=alpha_samples.size
    ds=round(float(iters)/nens)
    dsarray = np.arange(0,alpha_samples.size,ds)
    
    alpha_samples3 = alpha_samples[dsarray]
    #print('mean of sliced alpha_samples {}'.format(np.mean(alpha_samples)))
    beta_samples3 = beta_samples[dsarray]
    tau2_samples3 = tau2_samples[dsarray]
        
    # numpy empty storing tex data
    tex = np.empty((nd, nens))
    # estimate tex using given alpha_samples_comp, beta_samples_comp, and tau2_samples
    for i in range(nens):
        tau2_now  = tau2_samples3[i]
        beta_now  = beta_samples3[i]
        alpha_now = alpha_samples3[i]
        tex[:, i] = np.random.normal(seatemp * beta_now + alpha_now,
                                     np.sqrt(tau2_now))
    
    tex[tex > 1] = 1
    tex[tex < 0] = 0
    #grid_latlon = draws.find_nearest_latlon(lat=lat, lon=lon)  # useless?
    output = Prediction(ensemble=tex,
                        temptype=temptype,
                        analog_gridpoints=[tuple(latlon_match)])
    return output


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
    tex[tex > 1] = 1
    tex[tex < 0] = 0
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
