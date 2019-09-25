"""
Core
    Originated by Brewster Malevich

    Revised by: Mingsong Li
        Penn State
        Sept 23, 2019
    
    New file:   alpha_samples.mat
                beta_samples.mat
    Purpose: add TEX_forward model for analog model of baysparpy
            simplify the code and save space
            These two files were trimmed and calculated 
            using the following matlab code in TEX_forward.m of BAYSPAR
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
    example:
        alpha_samples_comp=read_draws('alpha_samples_comp.mat', 'sst')
    """
    drawtype = drawtype.lower()
    varstr = TRANSLATE_VAR[drawtype]
    if flstr[0:10] == 'Data_Input':
        var_template = 'observations/Data_Input_SpatAg_{0}.mat'
        # example:  
        # resource_str = 'observations/Data_Input_SpatAg_{0}.mat'.format('SST')
        resource_str = var_template.format(varstr)
        var = get_matlab_resource(resource_str, squeeze_me=True)  # return a dict
        Data_Input = var['Data_Input'] # return a numpy
        # print(Data_Input.dtype) # to show dtype
        # data bayspar wants include: 
        # Data_Input['Locs']
        # Data_Input['Target_Stack']
        # Data_Input['Inds_Stack']
        return Data_Input
        
    else:    
        var_template = 'modelparams/Output_SpatAg_{0}/{1}'
        # example:  
        # resource_str = 'modelparams/Output_SpatAg_{0}/{1}'.format('SST', 'alpha_samples_comp.mat')
        resource_str = var_template.format(varstr, flstr)
        varstr_full = os.path.splitext(flstr)[0]
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

# NEW class Draws_analog by Mingsong Li PennState
@attr.s
class Draws_analog:
    """Spatially-aware modelparams draws
    """
    alpha_samples = attr.ib()
    beta_samples = attr.ib()
    data_input = attr.ib()
    tau2_samples = attr.ib()
    locs_comp = attr.ib()
    _half_grid_space = attr.ib(default=10)
    
    # number of big grids:
    #N_bg = data_input['Locs'][0,0].shape[0]
    #
    #for kk in range(0,N_bg):
    #    # find the SST obs corresponding to this index location:
    #    vals=Data_Input['Target_Stack'][Data_Input['Inds_Stack'] == kk]
    #    # if the mean of vals in the big grids within tolerance, add it to inder_g
    #    if np.mean(vals) >= (np.mean(t)-stol) && mean(vals)<=(mean(t)+stol)
    #        inder_g = [inder_g; kk]
    

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


draws_sst_analog = Draws_analog(alpha_samples=read_draws('alpha_samples.mat', 'sst'),
                                beta_samples=read_draws('beta_samples.mat', 'sst'),
                                data_input = read_draws('Data_Input_SpatAg_SST','sst'),
                                tau2_samples=read_draws('tau2_samples.mat', 'sst'),
                                locs_comp=read_draws('Locs_Comp.mat', 'sst'))


draws_subt_analog = Draws_analog(alpha_samples=read_draws('alpha_samples.mat', 'subt'),
                                 beta_samples=read_draws('beta_samples.mat', 'subt'),
                                 data_input = read_draws('Data_Input_SpatAg_subT','subt'),
                                 tau2_samples=read_draws('tau2_samples.mat', 'subt'),
                                 locs_comp=read_draws('Locs_Comp.mat', 'subt'))

def get_draws(drawtype):
    """Get Draws instance for a draw type
    """
    if drawtype == 'sst':
        return deepcopy(draws_sst)
    elif drawtype == 'subt':
        return deepcopy(draws_subt)

# for analog model of get_draws
def get_draws_analog(drawtype):
    """Get Draws instance for a draw type
    for the analog model"""
    if drawtype == 'sst':
        return deepcopy(draws_sst_analog)
    elif drawtype == 'subt':
        return deepcopy(draws_subt_analog)
    