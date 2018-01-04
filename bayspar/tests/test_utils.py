import pytest
import numpy as np

from bayspar.utils import target_timeseries_pred


def test_target_timeseries_pred():
    np.random.seed(123)
    goal = np.array([-10.549816,  -0.657204,  -4.163044, -12.341982,  -6.353174])
    proxy_ts = np.array([0.2831, 0.2856, 0.2832, 0.2854, 0.3081])
    alpha_now = 0.3584
    beta_now = 0.0054
    tau2_now = 0.0016
    prior_pars = {'mu': np.array([0.0535] * 5),
                  'inv_cov': np.eye(5) * 0.0278}
    victim = target_timeseries_pred(alpha_now, beta_now, tau2_now,
                                    proxy_ts, prior_pars)
    np.testing.assert_allclose(victim, goal, atol=1e-4)
