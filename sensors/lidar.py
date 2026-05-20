import numpy as np


class LiDAR:
    """2D LiDAR sensor with configurable beams, range, and Gaussian noise."""

    def __init__(self, n_beams=180, max_range=12.0, noise_std=0.05):
        self.n_beams   = n_beams
        self.max_range = max_range
        self.noise_std = noise_std
        self.angles    = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)

    def scan(self, robot_state, env):
        """Return (true_distances, noisy_distances) for current robot pose."""
        x, y, theta = robot_state
        beam_angles  = theta + self.angles
        true_dist    = env.scan_distances(x, y, beam_angles, self.max_range)

        mult  = env.noise_multiplier(x, y)
        noise = np.random.normal(0.0, self.noise_std * mult, self.n_beams)
        noisy = np.clip(true_dist + noise, 0.0, self.max_range)
        return true_dist, noisy

    def to_cartesian(self, robot_state, distances):
        """Convert polar scan to Cartesian (x, y) point cloud."""
        x, y, theta = robot_state
        angles = theta + self.angles
        px = x + distances * np.cos(angles)
        py = y + distances * np.sin(angles)
        return px, py

    def filter_distances(self, distances, threshold=0.95):
        """Zero out beams that reach max range (likely free space)."""
        return np.where(distances < self.max_range * threshold, distances, self.max_range)

    def cluster_obstacles(self, distances):
        """Group consecutive obstacle-hitting beams into clusters."""
        hits = distances < self.max_range * 0.95
        clusters, in_c, start = [], False, 0
        for i, h in enumerate(hits):
            if h and not in_c:
                start, in_c = i, True
            elif not h and in_c:
                clusters.append((start, i - 1))
                in_c = False
        if in_c:
            clusters.append((start, len(distances) - 1))
        return clusters
