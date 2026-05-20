import numpy as np


class Forklift:
    """Dynamic obstacle — moves back and forth along a horizontal corridor."""

    PATH_Y    = 16.5   # m
    PATH_XMIN =  6.0   # m
    PATH_XMAX = 26.0   # m
    SPEED     =  0.4   # m/s
    RADIUS    =  0.8   # m

    def __init__(self):
        self.x    = self.PATH_XMIN
        self.y    = self.PATH_Y
        self.r    = self.RADIUS
        self._dir = 1   # +1 → right, -1 → left

    def step(self, dt):
        self.x += self._dir * self.SPEED * dt
        if self.x >= self.PATH_XMAX:
            self.x    = self.PATH_XMAX
            self._dir = -1
        elif self.x <= self.PATH_XMIN:
            self.x    = self.PATH_XMIN
            self._dir = 1


class Environment:
    WIDTH = 50.0
    HEIGHT = 50.0

    def __init__(self):
        self.obstacles = []
        self.start = np.array([3.0, 3.0])
        self.waypoints = [
            np.array([14.0, 12.0]),
            np.array([28.0, 28.0]),
            np.array([45.0, 45.0]),
        ]
        self.waypoint_names = ["Kalite Kontrol", "Ambalaj", "Soğuk Depo"]
        self.forklift = Forklift()
        self._build()

    def step(self, dt):
        """Advance dynamic obstacles by one timestep."""
        self.forklift.step(dt)

    def _build(self):
        # (x, y, w, h, name, noise_zone) — rectangular obstacles
        for x, y, w, h, name, noise in [
            (4,  10, 6.0,  4.0, "Hamur Makinesi",    True),
            (14,  4, 5.0,  6.0, "Karıştırıcı",       True),
            (24,  8, 7.0,  5.0, "Fırın",             True),
            (36,  4, 6.0,  4.0, "Soğutma Ünitesi",   True),
            (5,  22, 5.0,  8.0, "Ambalaj Makinesi",  True),
            (20, 32, 6.0,  5.0, "Etiketleme",        True),
            (34, 24, 7.0,  6.0, "Paketleme",         True),
            (40, 36, 5.0,  5.0, "Soğuk Depo Kapısı", False),
            (9,  37, 8.0,  1.5, "Konveyör 1",        False),
            (28, 14, 1.5, 12.0, "Konveyör 2",        False),
            (16, 43, 12.0, 1.5, "Konveyör 3",        False),
        ]:
            self.obstacles.append(dict(
                type='rect', x=float(x), y=float(y),
                w=float(w), h=float(h), name=name, noise_zone=noise,
            ))

        # (x, y, r, name) — circular pillars
        for x, y, r, name in [
            (13, 19, 1.0, "Sütun A"),
            (26, 22, 1.0, "Sütun B"),
            (38, 17, 1.0, "Sütun C"),
            (19, 40, 1.0, "Sütun D"),
        ]:
            self.obstacles.append(dict(
                type='circle', x=float(x), y=float(y),
                r=float(r), name=name, noise_zone=False,
            ))

    # ------------------------------------------------------------------ #
    #  Collision & noise                                                   #
    # ------------------------------------------------------------------ #

    def is_collision(self, x, y, margin=0.4):
        if x < margin or x > self.WIDTH - margin or y < margin or y > self.HEIGHT - margin:
            return True
        for obs in self.obstacles:
            if obs['type'] == 'rect':
                if (obs['x'] - margin <= x <= obs['x'] + obs['w'] + margin and
                        obs['y'] - margin <= y <= obs['y'] + obs['h'] + margin):
                    return True
            elif np.hypot(x - obs['x'], y - obs['y']) <= obs['r'] + margin:
                return True
        if np.hypot(x - self.forklift.x, y - self.forklift.y) <= self.forklift.r + margin:
            return True
        return False

    def noise_multiplier(self, x, y):
        """Return 3.0 near heavy machinery, 1.0 elsewhere."""
        for obs in self.obstacles:
            if obs.get('noise_zone') and obs['type'] == 'rect':
                cx = obs['x'] + obs['w'] / 2
                cy = obs['y'] + obs['h'] / 2
                if np.hypot(x - cx, y - cy) < 6.0:
                    return 3.0
        return 1.0

    # ------------------------------------------------------------------ #
    #  Vectorised ray casting (all LiDAR beams at once)                   #
    # ------------------------------------------------------------------ #

    def scan_distances(self, ox, oy, angles, max_range=12.0):
        dx = np.cos(angles)
        dy = np.sin(angles)
        dist = np.full(len(angles), max_range)

        # Boundary walls
        for wx in (0.0, self.WIDTH):
            mask = np.abs(dx) > 1e-9
            t = np.where(mask, (wx - ox) / np.where(mask, dx, 1.0), np.inf)
            dist = np.minimum(dist, np.where(mask & (t > 1e-6), t, np.inf))

        for wy in (0.0, self.HEIGHT):
            mask = np.abs(dy) > 1e-9
            t = np.where(mask, (wy - oy) / np.where(mask, dy, 1.0), np.inf)
            dist = np.minimum(dist, np.where(mask & (t > 1e-6), t, np.inf))

        for obs in self.obstacles:
            if obs['type'] == 'rect':
                t = self._slab_rect(dx, dy, ox, oy, obs)
            else:
                t = self._slab_circle(dx, dy, ox, oy, obs)
            dist = np.minimum(dist, t)

        # Dynamic obstacle: forklift
        fk = {'x': self.forklift.x, 'y': self.forklift.y, 'r': self.forklift.r}
        dist = np.minimum(dist, self._slab_circle(dx, dy, ox, oy, fk))

        return np.clip(dist, 0.0, max_range)

    def _slab_rect(self, dx, dy, ox, oy, obs):
        x1, y1 = obs['x'], obs['y']
        x2, y2 = x1 + obs['w'], y1 + obs['h']

        par_x = np.abs(dx) <= 1e-9
        par_y = np.abs(dy) <= 1e-9
        out_x = par_x & ((ox < x1) | (ox > x2))
        out_y = par_y & ((oy < y1) | (oy > y2))

        sdx = np.where(par_x, 1.0, dx)
        sdy = np.where(par_y, 1.0, dy)

        tx1 = np.where(par_x, -np.inf, (x1 - ox) / sdx)
        tx2 = np.where(par_x,  np.inf, (x2 - ox) / sdx)
        ty1 = np.where(par_y, -np.inf, (y1 - oy) / sdy)
        ty2 = np.where(par_y,  np.inf, (y2 - oy) / sdy)

        t_enter = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
        t_exit  = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))

        hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit > 1e-6) & ~out_x & ~out_y
        return np.where(hit, np.where(t_enter > 1e-6, t_enter, t_exit), np.inf)

    def _slab_circle(self, dx, dy, ox, oy, obs):
        fx = ox - obs['x']
        fy = oy - obs['y']
        b    = 2.0 * (fx * dx + fy * dy)
        c    = fx**2 + fy**2 - obs['r']**2
        disc = b**2 - 4.0 * c
        valid = disc >= 0
        sq   = np.sqrt(np.maximum(disc, 0.0))
        t1   = np.where(valid, (-b - sq) / 2.0, np.inf)
        t2   = np.where(valid, (-b + sq) / 2.0, np.inf)
        return np.where(valid & (t1 > 1e-6), t1,
                        np.where(valid & (t2 > 1e-6), t2, np.inf))
