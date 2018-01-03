import pytest
import numpy as np

from bayspar.predict import predict_seatemp, predict_sst, predict_subt, predict_tex


def test_predict_tex_sst():
    # TODO(brews): Double check this. It is a very rough test.

    np.random.seed(123)

    proxy_ts = np.array([1, 15, 30])
    lat = -79.49700165
    lon = -18.699981690000016
    temptype = 'sst'
    save_ensemble = True
    nens = 15000

    goal = {'preds': np.array([[0.30417229, 0.387622, 0.47020701],
                               [0.43638587, 0.54384393, 0.64827802],
                               [0.5504807, 0.71117075, 0.86330653]]),
            'siteloc': (lat, lon),
            'gridloc': (-80, -10),
            'predsens': np.ones((3, nens))
            }

    victim = predict_tex(dats=proxy_ts, lat=lat, lon=lon, temptype=temptype,
                         save_ensemble=save_ensemble, nens=nens)

    np.testing.assert_allclose(victim['preds'], goal['preds'], atol=0.25)
    assert victim['siteloc'] == goal['siteloc']
    assert victim['gridloc'] == goal['gridloc']
    assert victim['predsens'].shape == goal['predsens'].shape


def test_predict_tex_subt():
    # TODO(brews): Need this test.
    pass


def test_predict_seatemp():
    np.random.seed(123)

    proxy_ts = np.array([0.2831, 0.2856, 0.2832, 0.2854, 0.3081])
    prior_std = 6
    lat = -64.8527
    lon = -64.2080
    temptype = 'subt'
    save_ensemble = True
    nens = 15000

    goal = {'preds': np.array([[-11.23432951, -5.01136252, 1.13921719],
                               [-11.14555545, -4.7805361, 1.3855262],
                               [-11.39125685, -4.99742997, 1.14324675],
                               [-11.21444389, -4.82280281, 1.3729362],
                               [-9.55533595, -3.25380571, 2.91106291]]),
            'siteloc': (lat, lon),
            'gridloc': (-60, -70),
            'priormean': -0.434658923291294,
            'priorstd': 6,
            'predsens': np.ones((5, nens))
            }

    victim = predict_seatemp(dats=proxy_ts, lat=lat, lon=lon,
                             prior_std=prior_std, temptype=temptype,
                             save_ensemble=save_ensemble, nens=nens)

    np.testing.assert_allclose(victim['preds'], goal['preds'], atol=1)
    assert victim['siteloc'] == goal['siteloc']
    assert victim['gridloc'] == goal['gridloc']
    np.testing.assert_allclose(victim['priormean'], goal['priormean'],
                               atol=1e-5)
    assert victim['priorstd'] == goal['priorstd']
    assert victim['predsens'].shape == goal['predsens'].shape


def test_predict_sst():
    np.random.seed(123)

    proxy_ts = np.array([0.2831, 0.2856, 0.2832, 0.2854, 0.3081])
    prior_std = 6
    lat = -64.8527
    lon = -64.2080
    save_ensemble = True
    nens = 15000

    goal = {'preds': np.array([[-11.23432951, -5.01136252, 1.13921719],
                               [-11.14555545, -4.7805361, 1.3855262],
                               [-11.39125685, -4.99742997, 1.14324675],
                               [-11.21444389, -4.82280281, 1.3729362],
                               [-9.55533595, -3.25380571, 2.91106291]]),
            'siteloc': (lat, lon),
            'gridloc': (-60, -70),
            'priormean': 0.0535,
            'priorstd': 6,
            'predsens': np.ones((5, nens))
            }

    victim = predict_sst(dats=proxy_ts, lat=lat, lon=lon, prior_std=prior_std,
                         save_ensemble=save_ensemble, nens=nens)

    np.testing.assert_allclose(victim['preds'], goal['preds'], atol=1)
    assert victim['siteloc'] == goal['siteloc']
    assert victim['gridloc'] == goal['gridloc']
    np.testing.assert_allclose(victim['priormean'], goal['priormean'],
                               atol=1e-5)
    assert victim['priorstd'] == goal['priorstd']
    assert victim['predsens'].shape == goal['predsens'].shape


def test_predict_subt():
    np.random.seed(123)

    proxy_ts = np.array([0.2831, 0.2856, 0.2832, 0.2854, 0.3081])
    prior_std = 6
    lat = -64.8527
    lon = -64.2080
    save_ensemble = True
    nens = 15000

    goal = {'preds': np.array([[-11.23432951, -5.01136252, 1.13921719],
                               [-11.14555545, -4.7805361, 1.3855262],
                               [-11.39125685, -4.99742997, 1.14324675],
                               [-11.21444389, -4.82280281, 1.3729362],
                               [-9.55533595, -3.25380571, 2.91106291]]),
            'siteloc': (lat, lon),
            'gridloc': (-60, -70),
            'priormean': -0.434658923291294,
            'priorstd': 6,
            'predsens': np.ones((5, nens))
            }

    victim = predict_subt(dats=proxy_ts, lat=lat, lon=lon, prior_std=prior_std,
                          save_ensemble=save_ensemble, nens=nens)

    np.testing.assert_allclose(victim['preds'], goal['preds'], atol=1)
    assert victim['siteloc'] == goal['siteloc']
    assert victim['gridloc'] == goal['gridloc']
    np.testing.assert_allclose(victim['priormean'], goal['priormean'],
                               atol=1e-5)
    assert victim['priorstd'] == goal['priorstd']
    assert victim['predsens'].shape == goal['predsens'].shape
