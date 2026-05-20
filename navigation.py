import numpy as np


class PotentialFieldNav:
    """
    Attractive force toward goal + repulsive forces from LiDAR readings.
    Includes stuck-detection with random escape perturbation.
    """

    K_ATT       = 1.5
    K_REP       = 1.2
    D0          = 2.5    # repulsion influence range (m)
    MAX_V       = 1.5    # m/s
    MAX_W       = 1.5    # rad/s
    STUCK_WIN   = 30     # steps to look back for stuck detection
    STUCK_DIST  = 0.12   # m — movement threshold for stuck detection

    def __init__(self):
        self._hist         = []
        self._escape_steps = 0

    def compute(self, state, goal, distances, angles):
        x, y, theta = state
        gx, gy      = goal

        # --- Attractive force ---
        dx_g  = gx - x
        dy_g  = gy - y
        d_goal = np.hypot(dx_g, dy_g)
        if d_goal > 1.0:
            fax = self.K_ATT * dx_g / d_goal
            fay = self.K_ATT * dy_g / d_goal
        else:
            fax = self.K_ATT * dx_g
            fay = self.K_ATT * dy_g

        # --- Repulsive forces (vectorised over LiDAR beams) ---
        mask = (distances > 0.05) & (distances < self.D0)
        frx = fry = 0.0
        if np.any(mask):
            d   = distances[mask]
            ang = theta + angles[mask]
            ox  = x + d * np.cos(ang)
            oy  = y + d * np.sin(ang)
            ex  = x - ox
            ey  = y - oy
            mag = np.hypot(ex, ey) + 1e-6
            rep = self.K_REP * (1.0 / d - 1.0 / self.D0) / d**2
            frx = float(np.sum(rep * ex / mag))
            fry = float(np.sum(rep * ey / mag))

        ftx = fax + frx
        fty = fay + fry

        desired_angle = np.arctan2(fty, ftx)
        angle_err     = np.arctan2(np.sin(desired_angle - theta),
                                   np.cos(desired_angle - theta))
        f_mag = np.hypot(ftx, fty)
        v     = float(np.clip(f_mag * 0.5, 0.0, self.MAX_V))
        omega = float(np.clip(2.5 * angle_err, -self.MAX_W, self.MAX_W))

        # Slow down on sharp turns
        if abs(angle_err) > np.pi / 4:
            v *= 0.3

        # --- Stuck detection ---
        self._hist.append((x, y))
        if len(self._hist) > self.STUCK_WIN:
            self._hist.pop(0)
        if len(self._hist) == self.STUCK_WIN:
            moved = np.hypot(x - self._hist[0][0], y - self._hist[0][1])
            if moved < self.STUCK_DIST and self._escape_steps == 0:
                self._escape_steps = 25

        if self._escape_steps > 0:
            sign   = 1 if self._escape_steps % 2 == 0 else -1
            omega  = float(np.clip(sign * np.random.uniform(1.0, 1.8),
                                   -self.MAX_W, self.MAX_W))
            v      = 0.4
            self._escape_steps -= 1

        return v, omega


class Bug2Nav:
    """
    Bug2 algorithm: move toward goal along M-line; when obstacle hit,
    follow the boundary until the M-line is re-crossed closer to goal.
    """

    MAX_V         = 1.2
    MAX_W         = 1.5
    OBS_THRESHOLD = 1.3   # m — front obstacle triggers boundary mode
    WALL_DIST     = 1.5   # m — desired distance to right-side wall
    WALL_TIMEOUT  = 120   # steps — force GO_TO_GOAL if stuck in wall-follow
    M_LINE_TOL    = 1.8   # m — M-line crossing tolerance

    def __init__(self, start):
        self.start       = np.array(start, dtype=float)
        self._mode       = 'GO_TO_GOAL'
        self._hit_point  = None
        self._min_dist   = np.inf
        self._wall_steps = 0

    def reset(self, new_start):
        self.start       = np.array(new_start, dtype=float)
        self._mode       = 'GO_TO_GOAL'
        self._hit_point  = None
        self._min_dist   = np.inf
        self._wall_steps = 0

    def compute(self, state, goal, distances, angles):
        x, y, theta = state
        gx, gy      = goal
        d_goal      = np.hypot(gx - x, gy - y)

        # Front sector ±20° around heading
        n           = len(distances)
        front_idx   = list(range(n - n // 18, n)) + list(range(0, n // 18))
        front_min   = float(np.min([distances[i] for i in front_idx]))

        if self._mode == 'GO_TO_GOAL':
            self._wall_steps = 0
            if front_min < self.OBS_THRESHOLD:
                self._mode      = 'FOLLOW_WALL'
                self._hit_point = np.array([x, y])
                self._min_dist  = d_goal
            return self._go_to_goal(theta, gx - x, gy - y, d_goal)

        # FOLLOW_WALL
        self._wall_steps  += 1
        self._min_dist     = min(self._min_dist, d_goal)

        # Exit condition 1: M-line re-crossed and closer to goal
        m_line_exit = (front_min > self.OBS_THRESHOLD + 0.4 and
                       d_goal < self._min_dist - 0.2 and
                       self._on_m_line(x, y, goal))
        # Exit condition 2: timeout — robot stuck too long in boundary mode
        timeout_exit = self._wall_steps > self.WALL_TIMEOUT

        if m_line_exit or timeout_exit:
            self._mode       = 'GO_TO_GOAL'
            self._wall_steps = 0
            return self._go_to_goal(theta, gx - x, gy - y, d_goal)

        return self._follow_wall(distances, n)

    # ------------------------------------------------------------------ #

    def _go_to_goal(self, theta, dx, dy, d_goal):
        goal_angle = np.arctan2(dy, dx)
        err        = np.arctan2(np.sin(goal_angle - theta),
                                np.cos(goal_angle - theta))
        v     = float(np.clip(min(self.MAX_V, d_goal * 0.5), 0.0, self.MAX_V))
        omega = float(np.clip(2.0 * err, -self.MAX_W, self.MAX_W))
        if abs(err) > np.pi / 3:
            v *= 0.2
        return v, omega

    def _follow_wall(self, distances, n):
        # Right-side sector: 60–120° clockwise from forward (240–300° in array)
        r_start = int(n * 2 / 3)
        r_end   = int(n * 5 / 6)
        right   = float(np.min(distances[r_start:r_end]))
        err     = right - self.WALL_DIST
        v       = 0.7
        omega   = float(np.clip(-err * 1.5, -self.MAX_W, self.MAX_W))
        return v, omega

    def _on_m_line(self, x, y, goal):
        gx, gy   = goal
        sx, sy   = self.start
        length   = np.hypot(gx - sx, gy - sy)
        if length < 0.01:
            return True
        dist = abs((gy - sy) * x - (gx - sx) * y + gx * sy - gy * sx) / length
        return dist < self.M_LINE_TOL
