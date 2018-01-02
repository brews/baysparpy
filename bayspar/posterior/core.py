import os.path
import numpy as np
from scipy.io import loadmat


HERE = os.path.abspath(os.path.dirname(__file__))
SUBT_PATH = os.path.join(HERE, 'Output_SpatAg_SST')
SST_PATH = os.path.join(HERE, 'Output_SpatAg_subT')


def get_draws(flstr, drawtype):
    """Return single squeezed array from MATLAB draw file"""
    flpath = None
    if drawtype.lower() == 'sst':
        flpath = os.path.join(SST_PATH, flstr)
    elif drawtype.lower() == 'subt':
        flpath = os.path.join(SUBT_PATH, flstr)
    var_str = os.path.splitext(flstr)[0]
    return loadmat(flpath, squeeze_me=True)[var_str]


class Draws:
    def __init__(self, alpha_samples_comp, beta_samples_comp, tau2_samples, locs_comp):
        self.alpha_samples_comp = np.array(alpha_samples_comp)
        self.beta_samples_comp = np.array(beta_samples_comp)
        self.tau2_samples = np.array(tau2_samples)
        self.locs_comp = np.array(locs_comp)


sst_draws = Draws(alpha_samples_comp=get_draws('alpha_samples_comp.mat', 'sst'),
                  beta_samples_comp=get_draws('beta_samples_comp.mat', 'sst'),
                  tau2_samples=get_draws('tau2_samples.mat', 'sst'),
                  locs_comp=get_draws('Locs_Comp.mat', 'sst'))

subt_draws = Draws(alpha_samples_comp=get_draws('alpha_samples_comp.mat', 'subt'),
                   beta_samples_comp=get_draws('beta_samples_comp.mat', 'subt'),
                   tau2_samples=get_draws('tau2_samples.mat', 'subt'),
                   locs_comp=get_draws('Locs_Comp.mat', 'subt'))
