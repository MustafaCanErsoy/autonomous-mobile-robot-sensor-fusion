import numpy as np


class ExtendedKalmanFilter:
    """
    EKF for a differential-drive robot with state [x, y, theta].

    Predict : uses wheel-encoder (v, omega) as control input via the
              nonlinear kinematic model; linearises with Jacobian F.
    Update 1: corrects theta drift using IMU angular-velocity measurement.
              z = theta_prev + omega_imu * dt  (virtual theta observation).
    Update 2: absolute heading correction from magnetometer.
              z = theta_mag  (direct, drift-free but noisier).
    """

    def __init__(self, initial_state, P0=None, Q=None, R_imu=None, R_mag=None):
        self.state  = np.array(initial_state, dtype=float)   # [x, y, theta]
        self.P      = P0    if P0    is not None else np.diag([0.1, 0.1, 0.05])
        self.Q      = Q     if Q     is not None else np.diag([0.005, 0.005, 0.002])
        self.R_imu  = R_imu if R_imu is not None else np.array([[0.003]])
        self.R_mag  = R_mag if R_mag is not None else np.array([[0.008]])

    # ------------------------------------------------------------------ #
    #  Prediction step                                                     #
    # ------------------------------------------------------------------ #

    def predict(self, v, omega, dt):
        x, y, theta = self.state

        # Nonlinear state transition
        x_new     = x + v * np.cos(theta) * dt
        y_new     = y + v * np.sin(theta) * dt
        theta_new = np.arctan2(np.sin(theta + omega * dt),
                               np.cos(theta + omega * dt))
        self.state = np.array([x_new, y_new, theta_new])

        # Jacobian of f w.r.t. state (linearisation)
        F = np.array([
            [1.0, 0.0, -v * np.sin(theta) * dt],
            [0.0, 1.0,  v * np.cos(theta) * dt],
            [0.0, 0.0,  1.0],
        ])
        self.P = F @ self.P @ F.T + self.Q

    # ------------------------------------------------------------------ #
    #  Update step — generic                                              #
    # ------------------------------------------------------------------ #

    def update(self, z, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        innov = np.atleast_1d(z) - H @ self.state
        # Normalise angle innovation if theta is being observed
        if H.shape[0] == 1 and H[0, 2] == 1.0:
            innov[0] = np.arctan2(np.sin(innov[0]), np.cos(innov[0]))

        self.state = self.state + K.flatten() * innov.flatten()
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        self.P = (np.eye(3) - K @ H) @ self.P
        self.P = (self.P + self.P.T) / 2.0   # enforce symmetry against float drift
        return float(innov[0])                # pre-fit innovation (for logging)

    # ------------------------------------------------------------------ #
    #  Convenience: IMU theta update                                      #
    # ------------------------------------------------------------------ #

    def update_theta(self, z_theta, R=None):
        """Update with a virtual theta measurement from IMU integration."""
        H = np.array([[0.0, 0.0, 1.0]])
        return self.update(np.array([z_theta]), H, R if R is not None else self.R_imu)

    def update_theta_mag(self, z_theta, R=None):
        """Update with a direct absolute heading from magnetometer."""
        H = np.array([[0.0, 0.0, 1.0]])
        return self.update(np.array([z_theta]), H, R if R is not None else self.R_mag)

    def get_xy_covariance(self):
        """Return the 2×2 [x, y] covariance submatrix."""
        return self.P[:2, :2].copy()
