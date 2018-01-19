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
               transform=ccrs.Geodetic(), label=r'$TEX_{86}$ obs', zorder=3)
    return ax


def map_site(prediction, latlon=None, ax=None):
    """Plot a map of prediction site location
    """
    if latlon is not None:
        latlon = latlon
    else:
        latlon = prediction.location

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
            assert shp[1] >= n
            ax.plot(x, prediction.ensemble[:, -n:], alpha=0.05)
        elif len(shp) == 3:
            assert shp[1] * shp[2] >= n
            ax.plot(x, prediction.ensemble.reshape(shp[0], shp[1] * shp[2])[:, -n:],
                    alpha=0.1, color='black')

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    return ax


def analogmap(prediction, latlon=None, ax=None):
    """Map analog prediction with grids used for analog and TEX86 sites
    """
    if ax is None:
        ax = _default_map_ax()

    ax = map_texobs(obstype=prediction.temptype, ax=ax)

    if latlon is not None:
        ax = map_site(prediction, latlon=latlon, ax=ax)

    ax = map_analog_boxes(prediction, ax=ax)

    return ax
