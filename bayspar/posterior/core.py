import os.path
from copy import deepcopy
import numpy as np
from scipy.io import loadmat


HERE = os.path.abspath(os.path.dirname(__file__))
SUBT_PATH = os.path.join(HERE, 'Output_SpatAg_SST')
SST_PATH = os.path.join(HERE, 'Output_SpatAg_subT')


def read_draws(flstr, drawtype):
    """Grab single squeezed array from MATLAB draw file
    """
    flpath = None
    if drawtype.lower() == 'sst':
        flpath = os.path.join(SST_PATH, flstr)
    elif drawtype.lower() == 'subt':
        flpath = os.path.join(SUBT_PATH, flstr)
    variable = os.path.splitext(flstr)[0]
    return loadmat(flpath, squeeze_me=True)[variable]


class Draws:
    """Spatially-aware posterior draws
    """
    def __init__(self, alpha_samples_comp, alpha_samples_field,
                 beta_samples_comp, beta_samples_field, tau2_samples,
                 locs_comp):
        self._half_grid_space = 10
        self.alpha_samples_comp = np.array(alpha_samples_comp)
        self.alpha_samples_field = np.array(alpha_samples_field)
        self.beta_samples_comp = np.array(beta_samples_comp)
        self.beta_samples_field = np.array(beta_samples_field)
        self.tau2_samples = np.array(tau2_samples)
        self.locs_comp = np.array(locs_comp)

    def _index_near(self, lat, lon):
        """Get gridpoint index nearest a lat lon
        """
        assert -90 <= lat <= 90
        assert -180 < lon <= 180
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


draws_sst = Draws(alpha_samples_comp=read_draws('alpha_samples_comp.mat', 'sst'),
                  alpha_samples_field=read_draws('alpha_samples.mat', 'sst')['field'],
                  beta_samples_comp=read_draws('beta_samples_comp.mat', 'sst'),
                  beta_samples_field=read_draws('beta_samples.mat', 'sst')['field'],
                  tau2_samples=read_draws('tau2_samples.mat', 'sst'),
                  locs_comp=read_draws('Locs_Comp.mat', 'sst'))


draws_subt = Draws(alpha_samples_comp=read_draws('alpha_samples_comp.mat', 'subt'),
                   alpha_samples_field=read_draws('alpha_samples.mat', 'subt')['field'],
                   beta_samples_comp=read_draws('beta_samples_comp.mat', 'subt'),
                   beta_samples_field=read_draws('beta_samples.mat', 'subt')['field'],
                   tau2_samples=read_draws('tau2_samples.mat', 'subt'),
                   locs_comp=read_draws('Locs_Comp.mat', 'subt'))


def get_draws(drawtype):
    """Get Draws instance for a draw type
    """
    if drawtype == 'sst':
        return deepcopy(draws_sst)
    elif drawtype == 'subt':
        return deepcopy(draws_subt)
    else:
        assert drawtype in ['sst', 'subt']
