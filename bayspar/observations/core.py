import os.path
import numpy as np
from scipy.io import loadmat

HERE = os.path.abspath(os.path.dirname(__file__))


def get_obs(obstype):
    """Grab squeezed variable and locs array from flat MATLAB files
    """
    assert obstype.lower() in ['subt', 'sst']
    locs_template = 'locs_woa_1degree_asvec_{}.mat'
    var_template = 'st_woa_1degree_asvec_{}.mat'

    locs_path = None
    var_path = None
    if obstype == 'sst':
        locs_path = os.path.join(HERE, locs_template.format('SST'))
        var_path = os.path.join(HERE, var_template.format('SST'))
    elif obstype == 'subt':
        locs_path = os.path.join(HERE, locs_template.format('subT'))
        var_path = os.path.join(HERE, var_template.format('subT'))

    var = loadmat(var_path)['st_obs_ave_vec'].squeeze()
    locs = loadmat(locs_path)['locs_st_obs'].squeeze()

    return var, locs


def chord_distance(latlon1, latlon2):
    """Chordal distance between two groups of latlon points

    Parameters
    ----------
    latlon1 : ndarray
        An nx2 array of latitudes and longitudes for one set of points.
    latlon2 : ndarray
        An mx2 array of latitudes and longitudes for another set of points.

    Returns
    -------
    dists : 2d array
        An mxn array of Earth chordal distances [1]_ (km) between points in
        latlon1 and latlon2.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chord_(geometry)

    """
    earth_radius = 6378.137  # in km

    assert latlon1.shape[1] == 2
    assert latlon2.shape[1] == 2

    n = latlon1.shape[0]
    m = latlon2.shape[0]

    paired = np.hstack((np.kron(latlon1, np.ones((m, 1))),
                        np.kron(np.ones((n, 1)), latlon2)))

    latdif = np.deg2rad(paired[:, 0] - paired[:, 2])
    londif = np.deg2rad(paired[:, 1] - paired[:, 3])

    a = np.sin(latdif / 2) ** 2
    b = np.cos(np.deg2rad(paired[:, 0]))
    c = np.cos(np.deg2rad(paired[:, 2]))
    d = np.sin(np.abs(londif) / 2) ** 2

    half_angles = np.arcsin(np.sqrt(a + b * c * d))

    dists = 2 * earth_radius * np.sin(half_angles)

    return dists.reshape(m, n)


class ObsField:
    """Observed climate fields as used in calibration
    """
    def __init__(self, st_obs_ave_vec, locs_st_obs):
        self.st_obs_ave_vec = np.array(st_obs_ave_vec)
        self.locs_st_obs = np.array(locs_st_obs)

    def distance_from(self, lat, lon):
        """Chordal distance (km) of observations from latlon

        Parameters
        ----------
        lat : float or int
        lon : float or int

        Returns
        -------
        d : ndarray
            An nx1 array of distances (km) between latlon and the (n) observed
            points.
        """
        lat = float(lat)
        lon = float(lon)
        latlon = np.array([[lat, lon]])
        d = chord_distance(latlon, self.locs_st_obs[:, ::-1])
        return d

    def get_close_obs(self, lat, lon, distance=500, min_obs=1):
        """Get observations closest to a latlon point

        Parameters
        ----------
        lat : float or int
        lon : float or int
        distance : float or int
            Distance (km) of buffer from latlon point.
        min_obs : int
            Minimum number of obs to collect, if not enough within distance.

        Returns
        -------
        obs_sorted : ndarray
            1d array of observation values within buffer. Sorted by distance
            from latlon.
        d_sorted : ndarray
            Corresponding 1d array of observation distances (km) from latlon.
            Sorted by distance
        """
        lat = float(lat)
        lon = float(lon)
        distance = float(distance)
        min_obs = int(min_obs)

        d = self.distance_from(lat, lon)
        assert d.size == self.st_obs_ave_vec.size

        sort_idx = np.argsort(d.flat, axis=0)
        d_sorted = d.flat[sort_idx]
        obs_sorted = self.st_obs_ave_vec[sort_idx]

        n_in_buffer = (d < distance).sum()
        assert n_in_buffer > 0

        msk = np.empty(self.st_obs_ave_vec.shape, dtype=bool)
        msk.fill(0)
        if n_in_buffer > min_obs:
            msk = d_sorted < distance
        else:
            msk[:min_obs] = True

        return obs_sorted[msk], d_sorted[msk]


sst_obs = ObsField(*get_obs('sst'))

subt_obs = ObsField(*get_obs('subt'))
