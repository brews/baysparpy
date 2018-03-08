import os.path
from copy import deepcopy
from pkgutil import get_data
from io import BytesIO
import attr
import numpy as np
from scipy.io import loadmat


TRANSLATE_VAR = {'sst': 'SST', 'subt': 'subT'}


def get_matlab_resource(resource, package='bayspar', **kwargs):
    """Read flat MATLAB files as package resources, output for Numpy"""
    with BytesIO(get_data(package, resource)) as fl:
        data = loadmat(fl, **kwargs)
    return data


def read_draws(flstr, drawtype):
    """Grab single squeezed array from package resources
    """
    drawtype = drawtype.lower()

    var_template = 'modelparams/Output_SpatAg_{0}/{1}'

    varstr = TRANSLATE_VAR[drawtype]
    varstr_full = os.path.splitext(flstr)[0]
    resource_str = var_template.format(varstr, flstr)
    var = get_matlab_resource(resource_str, squeeze_me=True)
    return var[varstr_full]


@attr.s
class Draws:
    """Spatially-aware modelparams draws
    """
    alpha_samples_comp = attr.ib()
    beta_samples_comp = attr.ib()
    tau2_samples = attr.ib()
    locs_comp = attr.ib()
    _half_grid_space = attr.ib(default=10)

    def _index_near(self, lat, lon):
        """Get gridpoint index nearest a lat lon
        """
        if not (-90 <= lat <= 90) or not (-180 < lon <= 180):
            raise BadLatlonError(tuple([lat, lon]))

        lon_adiff = np.abs(self.locs_comp[:, 0] - lon) <= self._half_grid_space
        lat_adiff = np.abs(self.locs_comp[:, 1] - lat) <= self._half_grid_space
        return np.where(lon_adiff & lat_adiff)

    def find_nearest_latlon(self, lat, lon):
        """Find draws gridpoint nearest a given lat lon
        """
        idx = self._index_near(lat, lon)
        # Squeeze and flip so returns as latlon, not lonlat.
        latlon = self.locs_comp[idx].squeeze()[::-1]
        return latlon

    def find_alphabeta_near(self, lat, lon):
        """Find alpha and beta samples nearest a given location
        """
        idx = self._index_near(lat, lon)
        alpha_select = self.alpha_samples_comp[idx].squeeze()
        beta_select = self.beta_samples_comp[idx].squeeze()
        return alpha_select, beta_select


class BadLatlonError(Exception):
    """Raised when latitude or longitude is outside (-90, 90) or (-180, 180)

    Parameters
    ----------
    latlon : tuple
        The bad (lat, lon) values.
    """
    def __init__(self, latlon):
        self.latlon = latlon


draws_sst = Draws(alpha_samples_comp=read_draws('alpha_samples_comp.mat', 'sst'),
                  beta_samples_comp=read_draws('beta_samples_comp.mat', 'sst'),
                  tau2_samples=read_draws('tau2_samples.mat', 'sst'),
                  locs_comp=read_draws('Locs_Comp.mat', 'sst'))


draws_subt = Draws(alpha_samples_comp=read_draws('alpha_samples_comp.mat', 'subt'),
                   beta_samples_comp=read_draws('beta_samples_comp.mat', 'subt'),
                   tau2_samples=read_draws('tau2_samples.mat', 'subt'),
                   locs_comp=read_draws('Locs_Comp.mat', 'subt'))


def get_draws(drawtype):
    """Get Draws instance for a draw type
    """
    if drawtype == 'sst':
        return deepcopy(draws_sst)
    elif drawtype == 'subt':
        return deepcopy(draws_subt)
