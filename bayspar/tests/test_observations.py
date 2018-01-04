import pytest
import numpy as np

from bayspar.observations.core import chord_distance, sst_obs


def test_chord_distance():
    latlon1 = np.array([[25.5, 50], [30, -120], [1.5, 5]])
    latlon2 = np.array([[42.5, -122], [-15, -110.5], [-5.7, 3.5]])
    goal = np.array([[10550, 12542, 5877], [1400, 4976, 11140], [10771, 10758, 818]])

    victim = chord_distance(latlon1, latlon2)
    np.testing.assert_allclose(victim, goal, atol=1)


def test_distance_from():
    lat = 2
    lon = -100
    goal_head = 9010.52256915
    goal_tail = 8828.31316064
    goal_shape = (37686, 1)

    victim = sst_obs.distance_from(lat=lat, lon=lon)

    np.testing.assert_allclose(victim[0, 0], goal_head, atol=1)
    np.testing.assert_allclose(victim[-1, 0], goal_tail, atol=1)
    assert victim.shape == goal_shape


def test_get_close_obs():
    # TODO(brews): Need unittests to ensure the distance-min_obs switches work.
    lat = 2
    lon = -100

    goal_obs_head = 26.47
    goal_obs_tail = 23.64

    goal_dists_head = 78.68
    goal_dists_tail = 478.66

    goal_shape = (60,)

    victim_obs, victim_dists = sst_obs.get_close_obs(lat, lon)

    np.testing.assert_allclose(victim_obs[0], goal_obs_head, atol=1)
    np.testing.assert_allclose(victim_obs[-1], goal_obs_tail, atol=1)
    assert victim_obs.shape == goal_shape

    np.testing.assert_allclose(victim_dists[0], goal_dists_head, atol=1)
    np.testing.assert_allclose(victim_dists[-1], goal_dists_tail, atol=1)
    assert victim_dists.shape == goal_shape
