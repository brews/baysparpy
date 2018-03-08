import numpy as np
import scipy.stats as stats
import matplotlib.pylab as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from bayspar.observations import get_tex


def _default_map_ax(gca_kws=None, add_feature_kws=None):
    """Create default `matplotlib.Axes` instance for maps

    Parameters
    ----------
    gca_kws: dict, optional
        Keyword args passed to `matplotlib.gca()`.
    add_feature_kws: dict, optional
        Keyword args passed to `ax.addfeature()`.

    Returns
    -------
    ax: matplotlib.Axes instance.
    """
    if gca_kws is None:
        gca_kws = {'projection': ccrs.Robinson(central_longitude=0)}

    if add_feature_kws is None:
        add_feature_kws = {'feature': cfeature.LAND,
                           'facecolor': '#B0B0B0'}

    ax = plt.gca(**gca_kws)
    ax.add_feature(**add_feature_kws)

    return ax


def map_texobs(ax=None, texobs=None, obstype=None):
    """Plot a map of TEX86 observations
    """
    if texobs is None:
        texobs = get_tex(obstype)

    if ax is None:
        ax = _default_map_ax()

    ax.scatter(texobs.obslocs[:, 0], texobs.obslocs[:, 1], marker='.',
               transform=ccrs.Geodetic(), label=r'TEX$_{86}$ obs', zorder=3)
    return ax


def map_site(prediction, latlon=None, ax=None):
    """Plot a map of prediction site location
    """
    if latlon is not None:
        latlon = latlon
    else:
        latlon = prediction.latlon

    if ax is None:
        ax = _default_map_ax()

    ax.plot(latlon[1], latlon[0], marker='^', linestyle='None',
            transform=ccrs.Geodetic(), label='Prediction', color='C1', zorder=5)
    return ax


def map_analog_boxes(prediction, ax=None):
    """Plot map of grids used for analog prediction
    """
    if ax is None:
        ax = _default_map_ax()

    ys, xs = get_grid_corners(prediction.analog_gridpoints)
    for y, x in zip(ys, xs):
        ax.plot(x, y, transform=ccrs.PlateCarree(), color='black', zorder=4)

    return ax


def get_grid_corners(latlons, halfgrid=10):
    """Get 4 corners of observation grids

    Parameters
    ----------
    latlons : sequence
        Sequence of (lat, lon) for grid.
    halfgrid : int or float, optional
        Size of half-grid.

    Returns
    -------
    ys : list
        List containing 5-element tuples for latitude of each grid point.
    xs : list
        List containing 5-element tuples for longitude of each grid point.
    """
    ys = []
    xs = []
    for lat, lon in latlons:
        y = tuple([lat - halfgrid, lat - halfgrid,
                   lat + halfgrid, lat + halfgrid,
                   lat - halfgrid])
        x = tuple([lon - halfgrid, lon + halfgrid,
                   lon + halfgrid, lon - halfgrid,
                   lon - halfgrid])
        ys.append(y)
        xs.append(x)
    return ys, xs


def predictplot(prediction, ylabel=None, x=None, xlabel=None, spaghetti=False, ax=None):
    """Lineplot of prediction with uncertainty estimate

    Parameters
    ----------
    prediction : bayspar.predict.Prediction
        MCMC prediction
    ylabel : string, optional
        String label for y-axis.
    x : numpy.ndarray, optional
        Array over which to evaluate the densities. Default is
        `numpy.arange(0, 40.1, 0.1)`.
    xlabel : string, optional
        String label for x-axis.
    ax : matplotlib.Axes, optional
        Axes to plot onto.

    Returns
    -------
    ax : matplotlib.Axes
    """
    if ax is None:
        ax = plt.gca()

    if x is None:
        x = list(range(len(prediction.ensemble)))

    perc = prediction.percentile(q=[5, 50, 95])

    ax.fill_between(x, perc[:, 0], perc[:, 2], alpha=0.25,
                    label='90% uncertainty', color='C0')

    ax.plot(x, perc[:, 1], label='Median', color='C0')
    ax.plot(x, perc[:, 1], marker='.', color='C0')

    if prediction.prior_mean is not None:
        ax.axhline(prediction.prior_mean, label='Prior mean',
                   linestyle='dashed', color='C1')

    if spaghetti:
        n = 100
        shp = prediction.ensemble.shape
        if len(shp) == 2:
            ax.plot(x, prediction.ensemble[:, -n:], alpha=0.05)
        elif len(shp) == 3:
            ax.plot(x, prediction.ensemble.reshape(shp[0], shp[1] * shp[2])[:, -n:],
                    alpha=0.1, color='black')

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    return ax


def analogmap(prediction, latlon=None, ax=None):
    """Map analog prediction with grids used for analog and TEX86 sites

    Parameters
    ----------
    prediction : bayspar.predict.Prediction
        MCMC prediction
    latlon : sequence, optional
        Latitude and longitude for prediction site.
    ax : matplotlib.Axes, optional
        Axes to plot onto.

    Returns
    -------
    ax : matplotlib.Axes
    """
    if ax is None:
        ax = _default_map_ax()

    ax = map_texobs(obstype=prediction.temptype, ax=ax)

    if latlon is not None:
        ax = map_site(prediction, latlon=latlon, ax=ax)

    ax = map_analog_boxes(prediction, ax=ax)

    return ax


def densityplot(prediction, x=None, xlabel=None, ax=None):
    """Plot density of prediction prior and posterior

    Parameters
    ----------
    prediction : bayspar.predict.Prediction
        MCMC prediction
    x : numpy.ndarray, optional
        Array over which to evaluate the densities. Default is
        `numpy.arange(0, 40.1, 0.1)`.
    xlabel : string, optional
        String label for x-axis.
    ax : matplotlib.Axes, optional
        Axes to plot onto.

    Returns
    -------
    ax : matplotlib.Axes
    """
    if ax is None:
        ax = plt.gca()

    if x is None:
        x = np.arange(0, 40.1, 0.1)

    if prediction.prior_mean is not None and prediction.prior_std is not None:
        prior = stats.norm.pdf(x, prediction.prior_mean, prediction.prior_std)
        ax.plot(x, prior, color='C1', linestyle='dashed', label='Prior')

    kde = stats.gaussian_kde(prediction.ensemble.flat)
    post = kde(x)
    ax.plot(x, post, color='C0', label='Posterior')

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    return ax
