import numpy as np


class Magnetometer:
    """Absolute heading sensor — measures world-frame theta directly.

    Unlike the IMU (which integrates angular rate and accumulates drift),
    the magnetometer gives a drift-free but noisier theta observation.
    In EMI zones near heavy machinery the noise triples, matching the
    factory's electromagnetic interference model.
    """

    def __init__(self, theta_noise_std=0.05):
        self.theta_noise_std = theta_noise_std

    def measure(self, robot, env):
        """Return noisy absolute heading (rad), or None when in a blackout zone."""
        x, y, _ = robot.state
        if env.is_mag_blackout(x, y):
            return None
        mult  = env.noise_multiplier(x, y)
        noise = np.random.normal(0.0, self.theta_noise_std * mult)
        return np.arctan2(np.sin(robot.state[2] + noise),
                          np.cos(robot.state[2] + noise))
