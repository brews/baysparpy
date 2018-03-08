import pytest
import numpy as np

from bayspar.modelparams import get_draws
from bayspar.modelparams.core import BadLatlonError


def test__index_near():
    test_lat = -64.8527
    test_lon = -64.2080

    goal1 = 114
    goal2 = 110

    victim1 = get_draws('sst')._index_near(test_lat, test_lon)
    victim2 = get_draws('subt')._index_near(test_lat, test_lon)

    assert victim1[0] == goal1
    assert victim2[0] == goal2


def test__index_near_badlatlon():
    """Test _index_near raises if latlon outside of (-90, 90) or (-180, 180)
    """
    with pytest.raises(BadLatlonError):
        get_draws('sst')._index_near(45, 240)


def test_find_nearest_latlon():
    test_lat = -64.8527
    test_lon = -64.2080

    goal = np.array([-60, -70])

    victim1 = get_draws('sst').find_nearest_latlon(test_lat, test_lon)
    victim2 = get_draws('subt').find_nearest_latlon(test_lat, test_lon)

    np.testing.assert_equal(victim1, goal)
    np.testing.assert_equal(victim2, goal)
