import numpy as np


class IMU:
    """Inertial Measurement Unit — measures angular velocity with Gaussian noise."""

    def __init__(self, omega_noise_std=0.015):
        self.omega_noise_std = omega_noise_std

    def measure(self, robot, env):
        """Return noisy angular velocity (rad/s)."""
        x, y, _ = robot.state
        mult = env.noise_multiplier(x, y)
        return robot.omega + np.random.normal(0.0, self.omega_noise_std * mult)
