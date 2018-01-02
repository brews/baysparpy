import numpy as np

from bayspar.posterior import sst_draws, subt_draws


def predict_tex():
    pass


def predict_sst(analog=None):
    pass


def predict_subt(analog=None):
    pass


def predict_seatemp(dats, lon, lat, prior_std, temptype, nens=5000):
    assert temptype in ['sst', 'subt']
    assert -180 <= lon <= 180
    assert -90 <= lat <= 90
