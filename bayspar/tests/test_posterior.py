import pytest
import numpy as np

from bayspar.posterior import sst_draws, subt_draws


def test__index_near():
    test_lat = -64.8527
    test_lon = -64.2080

    goal1 = 110
    goal2 = 114

    victim1 = sst_draws._index_near(test_lat, test_lon)
    victim2 = subt_draws._index_near(test_lat, test_lon)

    assert victim1[0] == goal1
    assert victim2[0] == goal2


def test_find_nearest_latlon():
    test_lat = -64.8527
    test_lon = -64.2080

    goal = np.array([-60, -70])

    victim1 = sst_draws.find_nearest_latlon(test_lat, test_lon)
    victim2 = subt_draws.find_nearest_latlon(test_lat, test_lon)

    np.testing.assert_equal(victim1, goal)
    np.testing.assert_equal(victim2, goal)



