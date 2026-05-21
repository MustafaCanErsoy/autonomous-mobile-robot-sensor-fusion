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

    def median_filter_scan(self, distances, window=7):
        """Apply a circular 1-D median filter to remove noise spikes from beam distances."""
        n    = len(distances)
        half = window // 2
        out  = np.empty(n)
        for i in range(n):
            idx    = [(i + j - half) % n for j in range(window)]
            out[i] = np.median(distances[idx])
        return out

    def cluster_scan(self, distances, gap_threshold=1.5):
        """
        Gap-based sequential obstacle clustering on the LiDAR point cloud.

        Returns integer cluster labels (-1 = max-range / no-return beam).
        A new cluster starts whenever the Euclidean gap between consecutive
        valid hit-points exceeds gap_threshold (m).
        """
        mask   = distances < self.max_range * 0.95
        labels = np.full(len(distances), -1, dtype=int)
        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            return labels, 0

        cid = 0
        labels[valid_idx[0]] = 0
        px, py = self.to_cartesian(np.zeros(3), distances)   # robot at origin for relative
        for k in range(1, len(valid_idx)):
            i_prev = valid_idx[k - 1]
            i_curr = valid_idx[k]
            gap = np.hypot(px[i_curr] - px[i_prev], py[i_curr] - py[i_prev])
            if gap > gap_threshold or (i_curr - i_prev) > 5:
                cid += 1
            labels[i_curr] = cid
        return labels, cid + 1

