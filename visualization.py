import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

# ---- Global style -------------------------------------------------------
matplotlib.rcParams.update({
    'axes.titlesize':     14,
    'axes.titleweight':   'bold',
    'axes.labelsize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '#cccccc',
    'legend.borderpad':   0.55,
    'figure.facecolor':   'white',
    'savefig.facecolor':  'white',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   True,
    'axes.spines.bottom': True,
    'grid.alpha':         0.45,
    'grid.linewidth':     0.65,
    'grid.color':         '#d5d5d5',
    'lines.linewidth':    1.8,
    'patch.linewidth':    0.8,
})

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
#  Internal helpers                                                   #
# ------------------------------------------------------------------ #

def _add_arrows(ax, xs, ys, color, n_arrows=5, ms=14, lw=2.0, zorder=9):
    """Overlay evenly-spaced directional arrowheads along a path."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    total  = len(xs)
    if total < 6:
        return
    step   = max(total // 40, 2)
    idxs   = np.linspace(total // 8, 7 * total // 8, n_arrows).astype(int)
    for i in idxs:
        j = min(i + step, total - 1)
        dx, dy = xs[j] - xs[i], ys[j] - ys[i]
        if np.hypot(dx, dy) < 1e-6:
            continue
        ax.annotate('',
            xy=(xs[j], ys[j]), xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle='->', color=color,
                            lw=lw, mutation_scale=ms),
            zorder=zorder)

def _draw_env(ax, env, show_names=True):
    ax.set_xlim(0, env.WIDTH)
    ax.set_ylim(0, env.HEIGHT)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f4f8')
    ax.grid(True, alpha=0.55, linewidth=0.65, color='white', zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color('#888888')

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


def _draw_dynamic_routes(ax, env):
    """
    Draw dynamic-obstacle swept areas and bidirectional arrows on any axis.
    Makes it visually obvious why the robot detours around moving obstacles.
    """
    fk = env.forklift
    pl = env.palet

    # ---- Forklift (horizontal) ----------------------------------------
    # Swept area rectangle
    ax.add_patch(mpatches.Rectangle(
        (fk.PATH_XMIN, fk.PATH_Y - fk.RADIUS),
        fk.PATH_XMAX - fk.PATH_XMIN, 2 * fk.RADIUS,
        facecolor='#f39c12', alpha=0.20,
        edgecolor='#e67e22', linewidth=1.2, linestyle='--',
        zorder=2, label='Forklift hareket alanı'))
    # Bidirectional motion arrow
    ax.annotate('',
        xy=(fk.PATH_XMAX - 0.5, fk.PATH_Y),
        xytext=(fk.PATH_XMIN + 0.5, fk.PATH_Y),
        arrowprops=dict(arrowstyle='<->', color='#e67e22',
                        lw=1.8, mutation_scale=14),
        zorder=4)
    # Body circle at midpoint
    mid_fk = (fk.PATH_XMIN + fk.PATH_XMAX) / 2
    ax.add_patch(plt.Circle((mid_fk, fk.PATH_Y), fk.RADIUS,
                             color='#f39c12', alpha=0.70, zorder=5))
    ax.text(mid_fk, fk.PATH_Y, 'F', ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=6)

    # ---- Palet Robotu (vertical) ----------------------------------------
    ax.add_patch(mpatches.Rectangle(
        (pl.PATH_X - pl.RADIUS, pl.PATH_YMIN),
        2 * pl.RADIUS, pl.PATH_YMAX - pl.PATH_YMIN,
        facecolor='#9b59b6', alpha=0.20,
        edgecolor='#8e44ad', linewidth=1.2, linestyle='--',
        zorder=2, label='Palet Robotu hareket alanı'))
    ax.annotate('',
        xy=(pl.PATH_X, pl.PATH_YMAX - 0.5),
        xytext=(pl.PATH_X, pl.PATH_YMIN + 0.5),
        arrowprops=dict(arrowstyle='<->', color='#8e44ad',
                        lw=1.8, mutation_scale=14),
        zorder=4)
    mid_pl = (pl.PATH_YMIN + pl.PATH_YMAX) / 2
    ax.add_patch(plt.Circle((pl.PATH_X, mid_pl), pl.RADIUS,
                             color='#9b59b6', alpha=0.70, zorder=5))
    ax.text(pl.PATH_X, mid_pl, 'P', ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=6)


def _save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")


def _errors(true_path, est_path):
    n     = min(len(true_path), len(est_path))
    t_arr = np.array(true_path[:n])
    e_arr = np.array(est_path[:n])
    errs  = np.linalg.norm(t_arr[:, :2] - e_arr[:, :2], axis=1)
    return errs, float(np.sqrt(np.mean(errs**2))), float(np.mean(errs))


def _theta_errors(true_path, est_path):
    n     = min(len(true_path), len(est_path))
    t_arr = np.array(true_path[:n])
    e_arr = np.array(est_path[:n])
    diff  = t_arr[:, 2] - e_arr[:, 2]
    errs  = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
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
    (fk_line,) = ax.plot([fk.PATH_XMIN, fk.PATH_XMAX], [fk.PATH_Y, fk.PATH_Y],
                         '--', color='#f39c12', lw=1.2, alpha=0.6, label='Forklift rotası')
    pl = env.palet
    (pl_line,) = ax.plot([pl.PATH_X, pl.PATH_X], [pl.PATH_YMIN, pl.PATH_YMAX],
                         '--', color='#9b59b6', lw=1.2, alpha=0.6, label='Palet Robotu rotası')
    ax.legend(handles=[red_p, blue_p, grey_p, mag_p, fk_line, pl_line],
              loc='upper left', fontsize=8)

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

    # Planned path: straight lines from start through each waypoint (ideal, obstacle-free)
    plan_pts = np.array([env.start] + list(env.waypoints))

    for ax, path, title, color in [
        (axes[0], pf_path,   'Potential Field Navigasyon', '#e74c3c'),
        (axes[1], bug2_path, 'Bug2 Navigasyon',            '#2980b9'),
    ]:
        _draw_env(ax, env, show_names=False)
        _draw_dynamic_routes(ax, env)
        p = np.array(path)
        # Planned (ideal) path first so actual path renders on top
        ax.plot(plan_pts[:, 0], plan_pts[:, 1], 'k--', lw=2.2, alpha=0.50,
                zorder=3, label='Planlanan Yol (Düz Hat)')
        ax.plot(p[:, 0], p[:, 1], color=color, lw=2.0,
                label='Gerçek Robot Yolu', zorder=7, solid_capstyle='round')
        _add_arrows(ax, p[:, 0], p[:, 1], color, n_arrows=5, ms=13)
        ax.plot(p[0, 0], p[0, 1], 'o', ms=10, color='#27ae60',
                zorder=9, markeredgecolor='white', markeredgewidth=1.2)
        ax.plot(p[-1, 0], p[-1, 1], 's', ms=10, color='#8e44ad',
                zorder=9, label='Son Konum',
                markeredgecolor='white', markeredgewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(fontsize=8, loc='upper left')

    plt.suptitle('Yol Karşılaştırması: Potential Field vs Bug2',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, '02_path_comparison.png')
    plt.close(fig)


# ------------------------------------------------------------------ #
#  3. LiDAR sensor data                                               #
# ------------------------------------------------------------------ #

def plot_lidar(robot_state, true_dist, noisy_dist, lidar, env):
    filtered_dist = lidar.median_filter_scan(noisy_dist, window=7)
    labels, n_clusters = lidar.cluster_scan(filtered_dist, gap_threshold=1.5)

    x, y, theta = robot_state
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---- Left panel: raw noisy scan ----
    ax = axes[0]
    _draw_env(ax, env, show_names=False)
    px_raw, py_raw = lidar.to_cartesian(robot_state, noisy_dist)
    ax.scatter(px_raw, py_raw, s=4, c='#e74c3c', alpha=0.65, zorder=4,
               label='Ham LiDAR Noktaları (Gürültülü)')
    stride = max(1, lidar.n_beams // 36)
    for i in range(0, lidar.n_beams, stride):
        ang = theta + lidar.angles[i]
        ax.plot([x, x + noisy_dist[i] * np.cos(ang)],
                [y, y + noisy_dist[i] * np.sin(ang)],
                color='cyan', lw=0.5, alpha=0.30)
    ax.plot(x, y, 'bo', ms=9, label='Robot', zorder=6)
    ax.set_title('Ham LiDAR Ölçümü (Gürültülü)', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize=9)

    # ---- Right panel: median-filtered + obstacle clustering ----
    ax = axes[1]
    _draw_env(ax, env, show_names=False)

    # True-dist reference (faint white/grey — shows ideal signal)
    px_true, py_true = lidar.to_cartesian(robot_state, true_dist)
    mask_true = true_dist < lidar.max_range * 0.95
    ax.scatter(px_true[mask_true], py_true[mask_true], s=3,
               c='#bdc3c7', alpha=0.35, zorder=2, label='Gerçek Mesafe (referans)')

    # Clustered filtered points
    cmap = plt.cm.tab10
    px_f, py_f = lidar.to_cartesian(robot_state, filtered_dist)
    for cid in range(n_clusters):
        cmask = labels == cid
        if not np.any(cmask):
            continue
        lbl = f'Küme {cid + 1}' if cid < 6 else '_nolegend_'
        ax.scatter(px_f[cmask], py_f[cmask], s=7,
                   color=cmap(cid % 10), alpha=0.80, zorder=4, label=lbl)

    # No-return beams (max-range)
    no_return = labels == -1
    if np.any(no_return):
        ax.scatter(px_f[no_return], py_f[no_return], s=2,
                   c='#7f8c8d', alpha=0.20, zorder=1, label='Max Menzil (isabet yok)')

    ax.plot(x, y, 'bo', ms=9, label='Robot', zorder=6)
    ax.set_title(
        f'Filtrelenmiş LiDAR + Engel Kümeleme\n'
        f'(Medyan filtre pencere=7  |  {n_clusters} küme tespit edildi)',
        fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize=7, ncol=2, loc='upper left')

    plt.suptitle('LiDAR Sensör Verisi: Ham Ölçüm vs Medyan Filtrelenmiş + Kümeleme',
                 fontsize=14, fontweight='bold')
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
    ax_2d.plot(true[:n, 0], true[:n, 1], '-',  lw=2.2, color='#27ae60',
               label='Gerçek (Ground Truth)', zorder=5, solid_capstyle='round')
    ax_2d.plot(ekf[:n, 0],  ekf[:n, 1],  '--', lw=1.8, color='#2980b9',
               label='EKF Tahmini', zorder=4)
    ax_2d.plot(dr[:n, 0],   dr[:n, 1],   ':',  lw=1.8, color='#e74c3c',
               label='Dead Reckoning', zorder=3)
    _add_arrows(ax_2d, true[:n, 0], true[:n, 1], '#27ae60', n_arrows=4, ms=12)
    _add_arrows(ax_2d, dr[:n, 0],   dr[:n, 1],   '#e74c3c', n_arrows=3,
                ms=10, lw=1.5)

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
                    t_i = i * dt
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
    th_ekf,   th_rmse_ekf, th_mae_ekf = _theta_errors(true_path, ekf_path)
    th_dr,    th_rmse_dr,  th_mae_dr  = _theta_errors(true_path, dr_path)

    n = min(len(errs_ekf), len(errs_dr))
    t = np.arange(n) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    # ---- Position error ----
    ax1.fill_between(t, errs_dr[:n],  alpha=0.12, color='#e74c3c', zorder=1)
    ax1.fill_between(t, errs_ekf[:n], alpha=0.18, color='#2980b9', zorder=2)
    ax1.plot(t, errs_dr[:n],  color='#e74c3c', lw=1.8, alpha=0.90, zorder=3,
             label=f'Dead Reckoning   RMSE={rmse_dr:.3f} m | MAE={mae_dr:.3f} m')
    ax1.plot(t, errs_ekf[:n], color='#2980b9', lw=2.0, alpha=0.95, zorder=4,
             label=f'EKF              RMSE={rmse_ekf:.3f} m | MAE={mae_ekf:.3f} m')
    ax1.axhline(rmse_dr,  color='#e74c3c', linestyle='--', lw=1.2, alpha=0.55, zorder=2)
    ax1.axhline(rmse_ekf, color='#2980b9', linestyle='--', lw=1.2, alpha=0.55, zorder=2)
    ax1.text(t[-1] * 0.02, rmse_dr  + 0.04,
             f'RMSE_DR = {rmse_dr:.3f} m', color='#c0392b', fontsize=9, fontweight='bold')
    ax1.text(t[-1] * 0.02, rmse_ekf + 0.04,
             f'RMSE_EKF = {rmse_ekf:.3f} m', color='#1a5276', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Konum Hatası (m)')
    ax1.set_title('Konum Hatası — XY Düzlemi')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.35)

    # ---- Heading (theta) error ----
    th_ekf_deg = np.degrees(th_ekf[:n])
    th_dr_deg  = np.degrees(th_dr[:n])
    ax2.fill_between(t, th_dr_deg,  alpha=0.12, color='#e74c3c', zorder=1)
    ax2.fill_between(t, th_ekf_deg, alpha=0.18, color='#2980b9', zorder=2)
    ax2.plot(t, th_dr_deg,  color='#e74c3c', lw=1.8, alpha=0.90, zorder=3,
             label=f'Dead Reckoning   RMSE={np.degrees(th_rmse_dr):.2f}° | MAE={np.degrees(th_mae_dr):.2f}°')
    ax2.plot(t, th_ekf_deg, color='#2980b9', lw=2.0, alpha=0.95, zorder=4,
             label=f'EKF              RMSE={np.degrees(th_rmse_ekf):.2f}° | MAE={np.degrees(th_mae_ekf):.2f}°')
    ax2.axhline(np.degrees(th_rmse_dr),  color='#e74c3c', linestyle='--', lw=1.2, alpha=0.55)
    ax2.axhline(np.degrees(th_rmse_ekf), color='#2980b9', linestyle='--', lw=1.2, alpha=0.55)
    ax2.set_ylabel('Yönelim Hatası (°)')
    ax2.set_xlabel('Zaman (s)')
    ax2.set_title('Yönelim (Theta) Hatası')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)

    fig.suptitle('Hata Analizi: Dead Reckoning vs Extended Kalman Filter',
                 fontsize=13, fontweight='bold')
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

    # Dynamic obstacle routes
    fk = env.forklift
    ax.plot([fk.PATH_XMIN, fk.PATH_XMAX], [fk.PATH_Y, fk.PATH_Y],
            'w--', lw=1.5, alpha=0.6, label='Forklift rotası')
    pl = env.palet
    ax.plot([pl.PATH_X, pl.PATH_X], [pl.PATH_YMIN, pl.PATH_YMAX],
            '--', color='#c39bd3', lw=1.5, alpha=0.6, label='Palet Robotu rotası')

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


# ------------------------------------------------------------------ #
#  7. EKF sensor innovations (pre-fit residuals)                      #
# ------------------------------------------------------------------ #

def plot_ekf_innovations(imu_innovations, mag_innovations, dt=0.1):
    """Plot pre-fit innovations for IMU and magnetometer theta updates.

    A well-tuned EKF produces zero-mean innovations with variance ≈ S = HPH^T+R.
    Systematic bias or growing magnitude indicates filter inconsistency.
    """
    n = len(imu_innovations)
    t = np.arange(n) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # ---- IMU innovations ----
    imu_arr = np.degrees(np.array(imu_innovations))
    ax1.plot(t, imu_arr, color='#85c1e9', lw=0.6, alpha=0.50, label='IMU innovasyonu (ham)')
    win = 30
    if n > win:
        roll     = np.convolve(imu_arr, np.ones(win) / win, mode='valid')
        roll_sq  = np.convolve(imu_arr**2, np.ones(win) / win, mode='valid')
        roll_std = np.sqrt(np.maximum(roll_sq - roll**2, 0))
        t_roll   = t[win - 1:]
        ax1.fill_between(t_roll, roll - roll_std, roll + roll_std,
                         color='#2980b9', alpha=0.18, zorder=1, label='±1σ bandı')
        ax1.plot(t_roll, roll, color='#1a5276', lw=2.2,
                 label=f'Hareketli ort. ({win} adım)', zorder=3)
    ax1.axhline(0, color='#2c3e50', lw=1.2, linestyle='--', alpha=0.5)
    imu_rmse_deg = float(np.sqrt(np.mean(imu_arr**2)))
    ax1.set_ylabel('İnovasyon (°)', fontsize=10)
    ax1.set_title(f'IMU Sanal-Theta İnovasyonu  —  RMSE = {imu_rmse_deg:.3f}°',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ---- Magnetometer innovations ----
    valid_t, valid_v = [], []
    blackout_spans   = []
    in_bo, bo_start  = False, 0.0

    for i, v in enumerate(mag_innovations):
        t_i = i * dt
        if v is None:
            if not in_bo:
                bo_start, in_bo = t_i, True
        else:
            if in_bo:
                blackout_spans.append((bo_start, t_i))
                in_bo = False
            valid_t.append(t_i)
            valid_v.append(np.degrees(v))
    if in_bo:
        blackout_spans.append((bo_start, n * dt))

    for (t0, t1) in blackout_spans:
        ax2.axvspan(t0, t1, color='#8e44ad', alpha=0.15, zorder=0)

    if valid_v:
        vt  = np.array(valid_t)
        vv  = np.array(valid_v)
        ax2.scatter(vt, vv, s=5, c='#8e44ad', alpha=0.55, zorder=2,
                    label='Pusula innovasyonu')
        # rolling mean on valid samples only
        if len(vv) > win:
            roll_mag = np.convolve(vv, np.ones(win) / win, mode='valid')
            ax2.plot(vt[win - 1:], roll_mag, color='#6c3483', lw=2.0,
                     label=f'Hareketli ort. ({win} adım)')
        mag_rmse_deg = float(np.sqrt(np.mean(vv**2)))
        title_suffix = f'RMSE = {mag_rmse_deg:.3f}°'
    else:
        title_suffix = 'veri yok'

    bo_patch = mpatches.Patch(color='#8e44ad', alpha=0.30, label='Sinyal kesintisi (blackout)')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles + [bo_patch], fontsize=9, loc='upper right')
    ax2.axhline(0, color='k', lw=1.0, linestyle='--', alpha=0.4)
    ax2.set_xlabel('Zaman (s)', fontsize=10)
    ax2.set_ylabel('İnovasyon (°)', fontsize=10)
    ax2.set_title(f'Pusula (Magnetometre) Theta İnovasyonu  —  {title_suffix}',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        'EKF Sensör İnovasyonları (Ön-Düzeltme Artıkları)\n'
        'Sıfır ortalama ve sabit varyans → filtre tutarlılığının kanıtı',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, '07_ekf_innovations.png')
    plt.close(fig)


# ------------------------------------------------------------------ #
#  8. Sensor ablation study                                           #
# ------------------------------------------------------------------ #

def plot_sensor_ablation(env, true_path, dr_path, ekf_imu_path, ekf_full_path, dt=0.1):
    """Compare localization accuracy as sensors are progressively added to the EKF."""
    errs_dr,   rmse_dr,   mae_dr   = _errors(true_path, dr_path)
    errs_imu,  rmse_imu,  mae_imu  = _errors(true_path, ekf_imu_path)
    errs_full, rmse_full, mae_full  = _errors(true_path, ekf_full_path)

    n = min(len(errs_dr), len(errs_imu), len(errs_full))
    t = np.arange(n) * dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Time-series ----
    ax1.plot(t, errs_dr[:n],   color='#e74c3c', lw=1.4, alpha=0.85,
             label=f'Dead Reckoning              RMSE = {rmse_dr:.3f} m')
    ax1.plot(t, errs_imu[:n],  color='#f39c12', lw=1.6, alpha=0.90,
             label=f'EKF: Enkoder + IMU          RMSE = {rmse_imu:.3f} m')
    ax1.plot(t, errs_full[:n], color='#2980b9', lw=1.8, alpha=0.95,
             label=f'EKF: Enkoder + IMU + Pusula RMSE = {rmse_full:.3f} m')
    ax1.set_xlabel('Zaman (s)', fontsize=10)
    ax1.set_ylabel('Konum Hatası (m)', fontsize=10)
    ax1.set_title('Zaman Serisi — Her Sensörün Katkısı', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.35)

    # ---- Bar chart RMSE ----
    configs = ['Dead\nReckoning', 'EKF\nEnkoder+IMU', 'EKF\nEnkoder\n+IMU+Pusula']
    rmses   = [rmse_dr,  rmse_imu,  rmse_full]
    maes    = [mae_dr,   mae_imu,   mae_full]
    colors  = ['#e74c3c', '#f39c12', '#2980b9']
    edge_c  = ['#c0392b', '#d68910', '#1a5276']
    bars = ax2.bar(configs, rmses, color=colors, alpha=0.80,
                   edgecolor=edge_c, linewidth=1.5, width=0.52)
    y_max = rmse_dr * 1.35
    for bar, rmse, mae, ec in zip(bars, rmses, maes, edge_c):
        # Label inside bar for the tall bar, above for short bars
        y_label = bar.get_height() + y_max * 0.02
        ax2.text(bar.get_x() + bar.get_width() / 2, y_label,
                 f'RMSE {rmse:.3f} m\nMAE  {mae:.3f} m',
                 ha='center', va='bottom', fontsize=8.5,
                 fontweight='bold', color=ec)
    # Improvement arrows with connector lines
    for i in range(1, len(rmses)):
        imp  = (1 - rmses[i] / rmses[i - 1]) * 100
        x_l  = i - 1 + 0.26
        x_r  = i     - 0.26
        y_br = max(rmses[i - 1], rmses[i]) + y_max * 0.14
        ax2.annotate('', xy=(x_r, y_br), xytext=(x_l, y_br),
                     arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.6))
        ax2.text((x_l + x_r) / 2, y_br + y_max * 0.02,
                 f'−{imp:.1f}%', ha='center', va='bottom',
                 fontsize=9, color='#1e8449', fontweight='bold')
    ax2.set_ylabel('RMSE (m)')
    ax2.set_title('RMSE Karşılaştırması')
    ax2.set_ylim(0, y_max)
    ax2.grid(True, alpha=0.40, axis='y')

    fig.suptitle('Sensör Ablasyon Analizi — Her Sensörün EKF\'e Katkısı',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, '08_sensor_ablation.png')
    plt.close(fig)


# ------------------------------------------------------------------ #
#  9. Monte Carlo robustness analysis                                 #
# ------------------------------------------------------------------ #

def plot_monte_carlo(mc_ekf_errors, mc_dr_errors, dt=0.1):
    """Show EKF vs Dead Reckoning performance across multiple random seeds."""
    # Pad all runs to the same length (edge-fill with last value)
    max_n   = max(len(e) for e in mc_ekf_errors)
    ekf_mat = np.array([np.pad(e, (0, max_n - len(e)), mode='edge')
                        for e in mc_ekf_errors])
    dr_mat  = np.array([np.pad(e, (0, max_n - len(e)), mode='edge')
                        for e in mc_dr_errors])
    t = np.arange(max_n) * dt

    ekf_mean = ekf_mat.mean(axis=0)
    ekf_std  = ekf_mat.std(axis=0)
    dr_mean  = dr_mat.mean(axis=0)
    dr_std   = dr_mat.std(axis=0)

    ekf_rmses = [float(np.sqrt(np.mean(e ** 2))) for e in mc_ekf_errors]
    dr_rmses  = [float(np.sqrt(np.mean(e ** 2))) for e in mc_dr_errors]
    n_seeds   = len(mc_ekf_errors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Time series: individual runs + mean ± std ----
    for i, (e_ekf, e_dr) in enumerate(zip(ekf_mat, dr_mat)):
        ax1.plot(t, e_dr,  color='#e74c3c', lw=0.7, alpha=0.20)
        ax1.plot(t, e_ekf, color='#2980b9', lw=0.7, alpha=0.20)

    ax1.fill_between(t, dr_mean - dr_std, dr_mean + dr_std,
                     color='#e74c3c', alpha=0.22, label='_nolegend_')
    ax1.fill_between(t, ekf_mean - ekf_std, ekf_mean + ekf_std,
                     color='#2980b9', alpha=0.28, label='_nolegend_')
    ax1.plot(t, dr_mean,  color='#c0392b', lw=2.5,
             label=f'DR ort. ± std  ({np.mean(dr_rmses):.3f}±{np.std(dr_rmses):.3f} m RMSE)')
    ax1.plot(t, ekf_mean, color='#1a5276', lw=2.5,
             label=f'EKF ort. ± std ({np.mean(ekf_rmses):.3f}±{np.std(ekf_rmses):.3f} m RMSE)')
    ax1.set_xlabel('Zaman (s)', fontsize=10)
    ax1.set_ylabel('Konum Hatası (m)', fontsize=10)
    ax1.set_title(f'Zaman Serisi — {n_seeds} Seed (ince: tek çalışma, kalın: ortalama)',
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.35)

    # ---- Box plot of per-run RMSE ----
    bp = ax2.boxplot(
        [dr_rmses, ekf_rmses],
        labels=['Dead Reckoning', 'EKF (3 sensör)'],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color='#2c3e50', linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=5, alpha=0.6),
    )
    bp['boxes'][0].set_facecolor('#f1948a')
    bp['boxes'][1].set_facecolor('#85c1e9')

    # Overlay individual seed points
    for i, (dr_r, ekf_r) in enumerate(zip(dr_rmses, ekf_rmses)):
        ax2.plot(1 + np.random.uniform(-0.08, 0.08), dr_r,
                 'o', color='#c0392b', ms=5, alpha=0.7)
        ax2.plot(2 + np.random.uniform(-0.08, 0.08), ekf_r,
                 'o', color='#1a5276', ms=5, alpha=0.7)

    ax2.set_ylabel('RMSE (m)', fontsize=10)
    ax2.set_title(f'RMSE Dağılımı — {n_seeds} Farklı Seed\n'
                  f'EKF tutarlı olarak DR\'yi geçiyor',
                  fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.35, axis='y')

    fig.suptitle('Monte Carlo Dayanıklılık Analizi — EKF vs Dead Reckoning',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, '09_monte_carlo.png')
    plt.close(fig)
