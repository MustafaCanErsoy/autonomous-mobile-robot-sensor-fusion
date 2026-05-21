import numpy as np


class DifferentialDriveRobot:
    """Non-holonomic differential drive robot model."""

    def __init__(self, x, y, theta, wheel_base=0.5):
        self.state = np.array([x, y, theta], dtype=float)
        self.L = wheel_base   # distance between wheels (m)
        self.v = 0.0          # linear velocity (m/s)
        self.omega = 0.0      # angular velocity (rad/s)

    def step(self, v, omega, dt, env=None):
        x, y, theta = self.state
        x_new     = x + v * np.cos(theta) * dt
        y_new     = y + v * np.sin(theta) * dt
        theta_new = np.arctan2(np.sin(theta + omega * dt),
                               np.cos(theta + omega * dt))

        collision = env is not None and env.is_collision(x_new, y_new, margin=0.4)
        if collision:
            x_new, y_new = x, y

        self.state = np.array([x_new, y_new, theta_new])
        self.v     = 0.0 if collision else v   # encoder reads 0 when blocked by wall
        self.omega = omega

    def wheel_velocities(self):
        """Return (v_right, v_left) from current (v, omega)."""
        v_r = self.v + self.omega * self.L / 2.0
        v_l = self.v - self.omega * self.L / 2.0
        return v_r, v_l
