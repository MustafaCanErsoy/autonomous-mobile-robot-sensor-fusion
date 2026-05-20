import numpy as np


class WheelEncoder:
    """Wheel encoder — measures individual wheel velocities with Gaussian noise."""

    def __init__(self, noise_std=0.03):
        self.noise_std = noise_std

    def measure(self, robot, env):
        """Return (v_measured, omega_measured) derived from noisy wheel speeds."""
        x, y, _ = robot.state
        mult = env.noise_multiplier(x, y)
        v_r, v_l = robot.wheel_velocities()
        v_r += np.random.normal(0.0, self.noise_std * mult)
        v_l += np.random.normal(0.0, self.noise_std * mult)
        v_meas     = (v_r + v_l) / 2.0
        omega_meas = (v_r - v_l) / robot.L
        return v_meas, omega_meas
