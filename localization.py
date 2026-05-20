import numpy as np


class DeadReckoning:
    """Pure odometry integration — accumulates drift over time."""

    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta], dtype=float)

    def step(self, v, omega, dt):
        x, y, theta = self.state
        x     += v * np.cos(theta) * dt
        y     += v * np.sin(theta) * dt
        theta  = np.arctan2(np.sin(theta + omega * dt),
                            np.cos(theta + omega * dt))
        self.state = np.array([x, y, theta])


def compute_errors(true_path, est_path):
    """
    Compute per-step Euclidean position errors plus RMSE and MAE.

    Returns
    -------
    errors : ndarray  — per-step error (m)
    rmse   : float
    mae    : float
    """
    n      = min(len(true_path), len(est_path))
    true   = np.array(true_path[:n])
    est    = np.array(est_path[:n])
    errors = np.linalg.norm(true[:, :2] - est[:, :2], axis=1)
    rmse   = float(np.sqrt(np.mean(errors**2)))
    mae    = float(np.mean(errors))
    return errors, rmse, mae
