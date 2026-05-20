import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
#  Internal helpers                                                   #
# ------------------------------------------------------------------ #

def _draw_env(ax, env, show_names=True):
    ax.set_xlim(0, env.WIDTH)
    ax.set_ylim(0, env.HEIGHT)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, alpha=0.25, linewidth=0.5)

    for obs in env.obstacles:
        if obs['type'] == 'rect':
            color = '#c0392b' if obs.get('noise_zone') else '#2980b9'
            rect  = mpatches.Rectangle(
                (obs['x'], obs['y']), obs['w'], obs['h'],
                linewidth=0.8, edgecolor='#222', facecolor=color, alpha=0.72)
            ax.add_patch(rect)
            if show_names:
                ax.text(obs['x'] + obs['w'] / 2, obs['y'] + obs['h'] / 2,
                        obs['name'], ha='center', va='center',
                        fontsize=5.5, fontweight='bold', color='white')
        else:
            circle = plt.Circle((obs['x'], obs['y']), obs['r'],
                                 color='#7f8c8d', alpha=0.85, zorder=3)
            ax.add_patch(circle)
            if show_names:
                ax.text(obs['x'], obs['y'] + obs['r'] + 0.4,
                        obs['name'], ha='center', va='bottom', fontsize=5)

    # Magnetometer blackout zones
    for obs in env.obstacles:
        if obs['name'] in env._MAG_BLACKOUT and obs['type'] == 'rect':
            cx = obs['x'] + obs['w'] / 2
            cy = obs['y'] + obs['h'] / 2
            zone = plt.Circle((cx, cy), env._MAG_BLACKOUT_R,
                               color='#8e44ad', alpha=0.10, zorder=1,
                               linestyle='--', linewidth=0.8, fill=True)
            ax.add_patch(zone)

    ax.plot(*env.start, 'o', ms=10, color='#27ae60', label='Başlangıç', zorder=6)

    colors_wp = ['#f39c12', '#e67e22', '#e74c3c']
    for i, (wp, name) in enumerate(zip(env.waypoints, env.waypoint_names)):
        c = colors_wp[min(i, len(colors_wp) - 1)]
        ax.plot(*wp, '*', ms=13, color=c, zorder=6)
        ax.text(wp[0] + 0.6, wp[1] + 0.5,
                f'WP{i+1}: {name}', fontsize=6.5, color=c, fontweight='bold')


def _save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")


def _errors(true_path, est_path):
    n     = min(len(true_path), len(est_path))
    t_arr = np.array(true_path[:n])
    e_arr = np.array(est_path[:n])
    errs  = np.linalg.norm(t_arr[:, :2] - e_arr[:, :2], axis=1)
    return errs, float(np.sqrt(np.mean(errs**2))), float(np.mean(errs))


def _draw_cov_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Draw a confidence ellipse from a 2x2 covariance matrix."""
    vals, vecs = np.linalg.eigh(cov)
    vals  = np.maximum(vals, 0)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w, h  = 2.0 * n_std * np.sqrt(vals)
    ell   = Ellipse(xy=mean, width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell


# ------------------------------------------------------------------ #
#  1. Environment map                                                 #
# ------------------------------------------------------------------ #

def plot_environment(env):
    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_env(ax, env)

    red_p  = mpatches.Patch(color='#c0392b', alpha=0.72, label='Makineler (Yüksek Gürültü Bölgesi)')
    blue_p = mpatches.Patch(color='#2980b9', alpha=0.72, label='Konveyörler / Ekipman')
    grey_p = mpatches.Patch(color='#7f8c8d', alpha=0.85, label='Fabrika Sütunları')
    mag_p  = mpatches.Patch(color='#8e44ad', alpha=0.30, label='Pusula Sinyal Kesintisi')
    # Forklift + Palet routes
    fk = env.forklift
    ax.plot([fk.PATH_XMIN, fk.PATH_XMAX], [fk.PATH_Y, fk.PATH_Y],
            '--', color='#f39c12', lw=1.2, alpha=0.6, label='Forklift rotası')
    pl = env.palet
    ax.plot([pl.PATH_X, pl.PATH_X], [pl.PATH_YMIN, pl.PATH_YMAX],
            '--', color='#9b59b6', lw=1.2, alpha=0.6, label='Palet Robotu rotası')
    ax.legend(handles=[red_p, blue_p, grey_p, mag_p], loc='upper left', fontsize=8)

    ax.set_title('FrostBot — Pizza Fabrikası 2D Ortam Haritası', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.tight_layout()
    _save(fig, '01_environment.png')
    plt.close(fig)


# ------------------------------------------------------------------ #
#  2. Path comparison                                                 #
# ------------------------------------------------------------------ #

def plot_paths(env, pf_path, bug2_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, path, title, color in [
        (axes[0], pf_path,   'Potential Field Navigasyon', '#e74c3c'),
        (axes[1], bug2_path, 'Bug2 Navigasyon',            '#2980b9'),
    ]:
        _draw_env(ax, env, show_names=False)
        p = np.array(path)
        ax.plot(p[:, 0], p[:, 1], color=color, lw=1.8, label='Robot Yolu', zorder=5)
        ax.plot(p[0, 0], p[0, 1], 'o', ms=9, color='#27ae60', zorder=7)
        ax.plot(p[-1, 0], p[-1, 1], 's', ms=9, color='#8e44ad',
                zorder=7, label='Son Konum')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(fontsize=9)

    plt.suptitle('Yol Karşılaştırması: Potential Field vs Bug2',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, '02_path_comparison.png')
    plt.close(fig)


# ------------------------------------------------------------------ #
#  3. LiDAR sensor data                                               #
# ------------------------------------------------------------------ #

def plot_lidar(robot_state, true_dist, noisy_dist, lidar, env):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, dists, title, pt_color in [
        (axes[0], true_dist,  'Gerçek Mesafe (Gürültüsüz)',       '#27ae60'),
        (axes[1], noisy_dist, 'Gürültülü LiDAR (Ham Ölçüm)',    '#e74c3c'),
    ]:
        _draw_env(ax, env, show_names=False)
        px, py = lidar.to_cartesian(robot_state, dists)
        ax.scatter(px, py, s=4, c=pt_color, alpha=0.65, zorder=4,
                   label='LiDAR Noktaları')

        x, y, theta = robot_state
        stride = max(1, lidar.n_beams // 36)
        for i in range(0, lidar.n_beams, stride):
            ang = theta + lidar.angles[i]
            ax.plot([x, x + dists[i] * np.cos(ang)],
                    [y, y + dists[i] * np.sin(ang)],
                    color='cyan', lw=0.5, alpha=0.35)

        ax.plot(x, y, 'bo', ms=9, label='Robot', zorder=6)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(fontsize=9)

    plt.suptitle('LiDAR Sensör Verisi', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, '03_lidar_data.png')
    plt.close(fig)


# ------------------------------------------------------------------ #
#  4. Localization — 2D path + EKF ellipses + x/y/θ time series      #
# ------------------------------------------------------------------ #

def plot_localization(env, true_path, ekf_path, dr_path,
                      cov_history=None, mag_history=None, dt=0.1):
    fig = plt.figure(figsize=(20, 10))
    gs  = fig.add_gridspec(3, 2, width_ratios=[1.3, 1],
                           hspace=0.55, wspace=0.35)
    ax_2d = fig.add_subplot(gs[:, 0])
    ax_x  = fig.add_subplot(gs[0, 1])
    ax_y  = fig.add_subplot(gs[1, 1])
    ax_th = fig.add_subplot(gs[2, 1])

    true = np.array(true_path)
    ekf  = np.array(ekf_path)
    dr   = np.array(dr_path)
    n    = min(len(true), len(ekf), len(dr))
    t    = np.arange(n) * dt

    # ---- 2D path ----
    _draw_env(ax_2d, env, show_names=False)
    ax_2d.plot(true[:n, 0], true[:n, 1], 'g-',  lw=2.0,
               label='Gerçek (Ground Truth)', zorder=5)
    ax_2d.plot(ekf[:n, 0],  ekf[:n, 1],  'b--', lw=1.5,
               label='EKF Tahmini', zorder=4)
    ax_2d.plot(dr[:n, 0],   dr[:n, 1],   'r:',  lw=1.5,
               label='Dead Reckoning', zorder=3)

    # EKF covariance ellipses (95% confidence, every 3rd stored point)
    if cov_history:
        for i, (step, pos, cov) in enumerate(cov_history):
            if i % 3 == 0:
                _draw_cov_ellipse(
                    ax_2d, pos, cov, n_std=2.0,
                    facecolor='#3498db', alpha=0.10,
                    edgecolor='#2980b9', linewidth=0.6, zorder=2)
        # Legend proxy
        ell_proxy = mpatches.Patch(facecolor='#3498db', alpha=0.35,
                                   edgecolor='#2980b9', label='EKF 95% güven elipsi')
        handles, labels = ax_2d.get_legend_handles_labels()
        ax_2d.legend(handles=handles + [ell_proxy], fontsize=8, loc='upper left')
    else:
        ax_2d.legend(fontsize=8)

    ax_2d.set_title('Lokalizasyon — 2D Yol + EKF Kovaryans Ellipsi',
                    fontsize=11, fontweight='bold')
    ax_2d.set_xlabel('X (m)')
    ax_2d.set_ylabel('Y (m)')

    # ---- Time series: x(t), y(t), θ(t) ----
    series = [
        (ax_x,  0, 'x (m)',   'x(t)'),
        (ax_y,  1, 'y (m)',   'y(t)'),
        (ax_th, 2, 'θ (rad)', 'θ(t)'),
    ]
    for ax, col, ylabel, title in series:
        d_true = true[:n, col]
        d_ekf  = ekf[:n, col]
        d_dr   = dr[:n, col]
        if col == 2:
            d_true = np.arctan2(np.sin(d_true), np.cos(d_true))
            d_ekf  = np.arctan2(np.sin(d_ekf),  np.cos(d_ekf))
            d_dr   = np.arctan2(np.sin(d_dr),   np.cos(d_dr))
            # Raw magnetometer readings — skip None (blackout) steps
            if mag_history:
                m = min(len(mag_history), n - 1)
                valid_t, valid_v, blackout_spans = [], [], []
                in_blackout, bo_start = False, 0
                for i, v in enumerate(mag_history[:m]):
                    t_i = (i + 1) * dt
                    if v is None:
                        if not in_blackout:
                            bo_start, in_blackout = t_i, True
                    else:
                        if in_blackout:
                            blackout_spans.append((bo_start, t_i))
                            in_blackout = False
                        valid_t.append(t_i)
                        valid_v.append(np.arctan2(np.sin(v), np.cos(v)))
                if in_blackout:
                    blackout_spans.append((bo_start, m * dt))
                for (t0, t1) in blackout_spans:
                    ax.axvspan(t0, t1, color='#8e44ad', alpha=0.12, zorder=0)
                if valid_t:
                    vt = np.array(valid_t)
                    vv = np.array(valid_v)
                    ax.scatter(vt[::4], vv[::4], s=4, c='#95a5a6',
                               alpha=0.45, zorder=1, label='Pusula (ham)')
        ax.plot(t, d_true, 'g-',  lw=1.8, label='Gerçek')
        ax.plot(t, d_ekf,  'b--', lw=1.5, label='EKF')
        ax.plot(t, d_dr,   'r:',  lw=1.5, label='DR')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    ax_th.set_xlabel('Zaman (s)')

    fig.suptitle(
        'Lokalizasyon Sonuçları: Ground Truth vs EKF (Enkoder+IMU+Pusula) vs Dead Reckoning',
        fontsize=12, fontweight='bold', y=1.01)
    _save(fig, '04_localization.png')
    plt.close(fig)


# ------------------------------------------------------------------ #
#  5. Error analysis                                                  #
# ------------------------------------------------------------------ #

def plot_errors(true_path, ekf_path, dr_path, dt=0.1):
    errs_ekf, rmse_ekf, mae_ekf = _errors(true_path, ekf_path)
    errs_dr,  rmse_dr,  mae_dr  = _errors(true_path, dr_path)

    n = min(len(errs_ekf), len(errs_dr))
    t = np.arange(n) * dt

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(t, errs_dr[:n],  color='#e74c3c', lw=1.5, alpha=0.85,
            label=f'Dead Reckoning   RMSE={rmse_dr:.3f} m | MAE={mae_dr:.3f} m')
    ax.plot(t, errs_ekf[:n], color='#2980b9', lw=1.8, alpha=0.90,
            label=f'EKF              RMSE={rmse_ekf:.3f} m | MAE={mae_ekf:.3f} m')

    ax.axhline(rmse_dr,  color='#e74c3c', linestyle=':', lw=1.0, alpha=0.6)
    ax.axhline(rmse_ekf, color='#2980b9', linestyle=':', lw=1.0, alpha=0.6)
    ax.text(t[-1] * 0.02, rmse_dr  + 0.05,
            f'RMSE_DR  = {rmse_dr:.3f} m', color='#e74c3c', fontsize=9)
    ax.text(t[-1] * 0.02, rmse_ekf + 0.05,
            f'RMSE_EKF = {rmse_ekf:.3f} m', color='#2980b9', fontsize=9)

    ax.set_title('Konum Hatası Analizi: Dead Reckoning vs Extended Kalman Filter',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Zaman (s)')
    ax.set_ylabel('Konum Hatası (m)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    _save(fig, '05_error_analysis.png')
    plt.close(fig)

    return rmse_ekf, rmse_dr, mae_ekf, mae_dr


# ------------------------------------------------------------------ #
#  6. LiDAR heatmap                                                   #
# ------------------------------------------------------------------ #

def plot_lidar_heatmap(env, heatmap):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Heatmap — transpose so x→col, y→row; origin='lower' aligns (0,0) bottom-left
    extent = [0, env.WIDTH, 0, env.HEIGHT]
    im = ax.imshow(heatmap.T, origin='lower', extent=extent,
                   cmap='inferno', alpha=0.88, aspect='equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('LiDAR Tarama Yoğunluğu (isabet sayısı)', fontsize=9)

    # Obstacle outlines on top
    for obs in env.obstacles:
        if obs['type'] == 'rect':
            rect = mpatches.Rectangle(
                (obs['x'], obs['y']), obs['w'], obs['h'],
                linewidth=1.2, edgecolor='white', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
        else:
            circ = plt.Circle((obs['x'], obs['y']), obs['r'],
                               color='white', fill=False, linewidth=1.2, alpha=0.7)
            ax.add_patch(circ)

    # Forklift route
    fk = env.forklift
    ax.plot([fk.PATH_XMIN, fk.PATH_XMAX], [fk.PATH_Y, fk.PATH_Y],
            'w--', lw=1.5, alpha=0.6, label='Forklift rotası')

    ax.plot(*env.start, 'o', ms=10, color='#2ecc71', zorder=6, label='Başlangıç')
    for i, wp in enumerate(env.waypoints):
        ax.plot(*wp, '*', ms=12, color='#f1c40f', zorder=6)

    ax.set_xlim(0, env.WIDTH)
    ax.set_ylim(0, env.HEIGHT)
    ax.set_title('LiDAR Tarama Isı Haritası\n(Kırmızı = Sık Tarama, Siyah = Hiç Taranmadı)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    _save(fig, '06_lidar_heatmap.png')
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Animation — robot body (circle + heading arrow) + LiDAR + forklift#
# ------------------------------------------------------------------ #

def create_animation(env, true_path, ekf_path, lidar_snapshots, lidar,
                     forklift_history=None, palet_history=None, dt=0.1):
    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_env(ax, env, show_names=False)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    true_arr = np.array(true_path)
    ekf_arr  = np.array(ekf_path)

    # Paths
    true_line, = ax.plot([], [], 'g-',  lw=1.5, label='Gerçek Yol',  zorder=4)
    ekf_line,  = ax.plot([], [], 'b--', lw=1.0, label='EKF Tahmini', zorder=3)

    # Robot body — circle drawn as closed parametric curve
    _circle_t  = np.linspace(0, 2 * np.pi, 24)
    R_body     = 0.6   # robot radius (m)
    robot_body, = ax.plot([], [], color='#27ae60', lw=2.0, zorder=7)
    robot_head, = ax.plot([], [], color='#27ae60', lw=2.5, zorder=8)  # heading line

    # EKF position marker
    ekf_dot, = ax.plot([], [], 'b^', ms=7, zorder=6, label='EKF Konum')

    # LiDAR point cloud
    lidar_sc = ax.scatter([], [], s=2, c='cyan', alpha=0.45, zorder=2, label='LiDAR')

    # Dynamic obstacle bodies (parametric circles)
    _dyn_t    = np.linspace(0, 2 * np.pi, 20)
    fk_body,  = ax.plot([], [], color='#f39c12', lw=2.5, zorder=7, label='Forklift')
    fk_dot,   = ax.plot([], [], 'o', color='#f39c12', ms=4, zorder=8)
    pl_body,  = ax.plot([], [], color='#9b59b6', lw=2.5, zorder=7, label='Palet Robotu')
    pl_dot,   = ax.plot([], [], 'o', color='#9b59b6', ms=4, zorder=8)

    # Static route markers
    fk = env.forklift
    ax.plot([fk.PATH_XMIN, fk.PATH_XMAX], [fk.PATH_Y, fk.PATH_Y],
            '--', color='#f39c12', lw=0.8, alpha=0.4)
    pl = env.palet
    ax.plot([pl.PATH_X, pl.PATH_X], [pl.PATH_YMIN, pl.PATH_YMAX],
            '--', color='#9b59b6', lw=0.8, alpha=0.4)

    # Time label
    time_txt = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top',
                       fontsize=9, color='#2c3e50',
                       bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_title('FrostBot — Pizza Fabrikası Simülasyonu', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)

    snap_dict = {s: nd for s, _td, nd in lidar_snapshots}
    N         = len(true_path)
    stride    = max(1, N // 180)
    frames    = list(range(0, N, stride))

    def _update(fi):
        step        = frames[fi]
        x, y, theta = true_arr[step]

        # Paths
        true_line.set_data(true_arr[:step+1, 0], true_arr[:step+1, 1])
        ekf_line.set_data(ekf_arr[:step+1, 0],   ekf_arr[:step+1, 1])

        # Robot body (circle)
        bx = x + R_body * np.cos(_circle_t)
        by = y + R_body * np.sin(_circle_t)
        robot_body.set_data(bx, by)

        # Heading arrow (line from centre to front)
        hx = [x, x + R_body * 1.6 * np.cos(theta)]
        hy = [y, y + R_body * 1.6 * np.sin(theta)]
        robot_head.set_data(hx, hy)

        # EKF marker
        ekf_dot.set_data([ekf_arr[step, 0]], [ekf_arr[step, 1]])

        # LiDAR
        nearest = min(snap_dict.keys(), key=lambda s: abs(s - step))
        px, py  = lidar.to_cartesian(true_arr[step], snap_dict[nearest])
        lidar_sc.set_offsets(np.c_[px, py])

        # Forklift
        if forklift_history and step < len(forklift_history):
            fkx, fky = forklift_history[step]
            fk_body.set_data(fkx + fk.RADIUS * np.cos(_dyn_t),
                             fky + fk.RADIUS * np.sin(_dyn_t))
            fk_dot.set_data([fkx], [fky])

        # Palet Robotu
        if palet_history and step < len(palet_history):
            plx, ply = palet_history[step]
            pl_body.set_data(plx + pl.RADIUS * np.cos(_dyn_t),
                             ply + pl.RADIUS * np.sin(_dyn_t))
            pl_dot.set_data([plx], [ply])

        time_txt.set_text(f't = {step * dt:.1f} s')
        return (true_line, ekf_line, robot_body, robot_head,
                ekf_dot, lidar_sc, fk_body, fk_dot, pl_body, pl_dot, time_txt)

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=50, blit=True)

    path = os.path.join(OUTPUT_DIR, 'animation.gif')
    anim.save(path, writer=animation.PillowWriter(fps=20))
    print(f"  Saved: {path}")
    plt.close(fig)
