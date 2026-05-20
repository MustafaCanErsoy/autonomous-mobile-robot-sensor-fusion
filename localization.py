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
