import pytest
import numpy as np

from bayspar.predict import (Prediction, predict_seatemp, predict_tex,
                             predict_seatemp_analog, EnsembleSizeError)


def test_percentile():
    prediction_test = Prediction(ensemble=np.array([range(10), range(10)]),
                                 temptype='sst')
    victim = prediction_test.percentile()
    goal = np.array([[0, 0], [4, 4], [9, 9]]).T
    np.testing.assert_equal(victim, goal)


def test_predict_tex_EnsembleSizeError():
    """Check raised error with very large nens from user."""
    proxy_ts = np.array([1, 15, 30])
    lat = -79.49700165
    lon = -18.699981690000016
    temptype = 'sst'
    nens = 15000

    with pytest.raises(EnsembleSizeError):
        predict_tex(seatemp=proxy_ts, lat=lat, lon=lon, temptype=temptype, nens=nens)


def test_predict_tex_sst():
    # TODO(brews): Double check this. It is a very rough test.

    np.random.seed(123)

    proxy_ts = np.array([1, 15, 30])
    lat = -79.49700165
    lon = -18.699981690000016
    temptype = 'sst'
    nens = 10000

    goal = {'preds': np.array([[0.30417229, 0.387622, 0.47020701],
                               [0.43638587, 0.54384393, 0.64827802],
                               [0.5504807, 0.71117075, 0.86330653]]),
            'siteloc': (lat, lon),
            'gridloc': [(-80, -10)],
            'predsens': np.ones((3, nens))
            }

    victim = predict_tex(seatemp=proxy_ts, lat=lat, lon=lon, temptype=temptype, nens=nens)

    np.testing.assert_allclose(victim.percentile(), goal['preds'], atol=0.25)
    assert victim.latlon == goal['siteloc']
    assert victim.modelparam_gridpoints == goal['gridloc']
    assert victim.ensemble.shape == goal['predsens'].shape


@pytest.mark.skip(reason='Not implemented')
def test_predict_tex_subt():
    # TODO(brews): Need this test.
    raise NotImplementedError


def test_predict_seatemp():
    np.random.seed(123)

    proxy_ts = np.array([0.2831, 0.2856, 0.2832, 0.2854, 0.3081])
    prior_std = 6
    lat = -64.8527
    lon = -64.2080
    temptype = 'subt'
    nens = 1000

    goal = {'preds': np.array([[-11.23432951, -5.01136252, 1.13921719],
                               [-11.14555545, -4.7805361, 1.3855262],
                               [-11.39125685, -4.99742997, 1.14324675],
                               [-11.21444389, -4.82280281, 1.3729362],
                               [-9.55533595, -3.25380571, 2.91106291]]),
            'siteloc': (lat, lon),
            'gridloc': [(-60, -70)],
            'priormean': -0.434658923291294,
            'priorstd': 6,
            'predsens': np.ones((5, nens))
            }

    victim = predict_seatemp(tex=proxy_ts, lat=lat, lon=lon, prior_std=prior_std, temptype=temptype, nens=nens)

    np.testing.assert_allclose(victim.percentile(), goal['preds'], atol=1)
    assert victim.latlon == goal['siteloc']
    assert victim.modelparam_gridpoints == goal['gridloc']
    np.testing.assert_allclose(victim.prior_mean, goal['priormean'],
                               atol=1e-5)
    assert victim.prior_std == goal['priorstd']
    assert victim.ensemble.shape == goal['predsens'].shape


def test_predict_seatemp_analog_sst():
    np.random.seed(123)

    proxy_ts = np.array([0.7900, 0.7400, 0.7700, 0.7000])
    temptype = 'sst'
    prior_std = 20
    prior_mean = 30
    nens = 10000
    search_tol = np.std(proxy_ts, ddof=1) * 2

    goal = {'preds': np.array([[27.518104, 32.925663, 39.850845],
                               [24.606164, 29.9325, 36.152334],
                               [26.399232, 31.738569, 38.387674],
                               [22.099501, 27.518964, 33.240762]]),
            'anlocs': np.array([[110, 20], [110, 0], [50, 20], [90, 0],
                                [150, 0], [50, -20], [70, 20], [170, 0],
                                [-170, -20], [-90, 20], [-70, 20], [-170, 0],
                                [110, -20]]),
            'priormean': 30,
            'priorstd': 20,
            'predsens': np.ones((4, 13, nens))
            }

    victim = predict_seatemp_analog(tex=proxy_ts, prior_std=prior_std, temptype=temptype, search_tol=search_tol,
                                    prior_mean=prior_mean, nens=nens)

    np.testing.assert_allclose(victim.percentile(), goal['preds'], atol=1)

    # Because Im stupid and wrote lonlat array above.
    goal_anloc = [tuple(x) for x in goal['anlocs'][:, ::-1].tolist()]
    goal_anloc.sort()
    victim_anloc = victim.analog_gridpoints.copy()
    victim_anloc.sort()
    assert victim_anloc == goal_anloc

    assert victim.prior_mean == goal['priormean']
    assert victim.prior_std == goal['priorstd']
    assert victim.ensemble.shape == goal['predsens'].shape
