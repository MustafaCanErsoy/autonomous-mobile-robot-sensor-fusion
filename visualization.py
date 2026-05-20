import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

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

    # Start point
    ax.plot(*env.start, 'o', ms=10, color='#27ae60',
            label='Başlangıç', zorder=6)

    # Waypoints
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
    n      = min(len(true_path), len(est_path))
    t_arr  = np.array(true_path[:n])
    e_arr  = np.array(est_path[:n])
    errs   = np.linalg.norm(t_arr[:, :2] - e_arr[:, :2], axis=1)
    return errs, float(np.sqrt(np.mean(errs**2))), float(np.mean(errs))


# ------------------------------------------------------------------ #
#  1. Environment map                                                 #
# ------------------------------------------------------------------ #

def plot_environment(env):
    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_env(ax, env)

    red_p  = mpatches.Patch(color='#c0392b', alpha=0.72, label='Makineler (Yüksek Gürültü Bölgesi)')
    blue_p = mpatches.Patch(color='#2980b9', alpha=0.72, label='Konveyörler / Ekipman')
    grey_p = mpatches.Patch(color='#7f8c8d', alpha=0.85, label='Fabrika Sütunları')
    ax.legend(handles=[red_p, blue_p, grey_p], loc='upper left', fontsize=8)

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
        ax.plot(p[:, 0], p[:, 1], color=color, lw=1.8,
                label='Robot Yolu', zorder=5)
        ax.plot(p[0, 0], p[0, 1], 'o', ms=9, color='#27ae60', zorder=7)
        ax.plot(p[-1, 0], p[-1, 1], 's', ms=9, color='#8e44ad', zorder=7,
                label='Son Konum')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(fontsize=9)

    plt.suptitle('Yol Karşılaştırması: Potential Field vs Bug2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, '02_path_comparison.png')
    plt.close(fig)


# ------------------------------------------------------------------ #
#  3. LiDAR sensor data                                               #
# ------------------------------------------------------------------ #

def plot_lidar(robot_state, true_dist, noisy_dist, lidar, env):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, dists, title, pt_color in [
        (axes[0], true_dist,  'Ham LiDAR (Gürültüsüz)',        '#27ae60'),
        (axes[1], noisy_dist, 'Filtrelenmiş LiDAR (Gürültülü)', '#e74c3c'),
    ]:
        _draw_env(ax, env, show_names=False)
        px, py = lidar.to_cartesian(robot_state, dists)
        ax.scatter(px, py, s=4, c=pt_color, alpha=0.65, zorder=4,
                   label='LiDAR Noktaları')

        x, y, theta = robot_state
        # Draw a subset of rays
        stride = max(1, lidar.n_beams // 36)
        for i in range(0, lidar.n_beams, stride):
            ang = theta + lidar.angles[i]
            rx  = x + dists[i] * np.cos(ang)
            ry  = y + dists[i] * np.sin(ang)
            ax.plot([x, rx], [y, ry], color='cyan', lw=0.5, alpha=0.35)

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
#  4. Localization results                                            #
# ------------------------------------------------------------------ #

def plot_localization(env, true_path, ekf_path, dr_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    true = np.array(true_path)
    ekf  = np.array(ekf_path)
    dr   = np.array(dr_path)
    n    = min(len(true), len(ekf), len(dr))
    t    = np.arange(n) * 0.1

    # 2D overlay
    ax = axes[0]
    _draw_env(ax, env, show_names=False)
    ax.plot(true[:n, 0], true[:n, 1], 'g-',  lw=2.0, label='Gerçek Konum (Ground Truth)', zorder=5)
    ax.plot(ekf[:n, 0],  ekf[:n, 1],  'b--', lw=1.5, label='EKF Tahmini',                zorder=4)
    ax.plot(dr[:n, 0],   dr[:n, 1],   'r:',  lw=1.5, label='Dead Reckoning',             zorder=3)
    ax.set_title('Lokalizasyon (2D Yol)', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize=9)

    # Time series x(t)
    ax = axes[1]
    ax.plot(t, true[:n, 0], 'g-',  lw=1.8, label='x Gerçek')
    ax.plot(t, ekf[:n, 0],  'b--', lw=1.5, label='x EKF')
    ax.plot(t, dr[:n, 0],   'r:',  lw=1.5, label='x Dead Reckoning')
    ax.set_title('Lokalizasyon — x(t) Zaman Serisi', fontsize=12, fontweight='bold')
    ax.set_xlabel('Zaman (s)')
    ax.set_ylabel('x Konumu (m)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    plt.suptitle('Lokalizasyon Sonuçları: Ground Truth vs EKF vs Dead Reckoning',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
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
    ax.text(t[-1] * 0.02, rmse_dr  + 0.02, f'RMSE_DR  = {rmse_dr:.3f} m',
            color='#e74c3c', fontsize=9)
    ax.text(t[-1] * 0.02, rmse_ekf + 0.02, f'RMSE_EKF = {rmse_ekf:.3f} m',
            color='#2980b9', fontsize=9)

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
#  Animation                                                          #
# ------------------------------------------------------------------ #

def create_animation(env, true_path, ekf_path, lidar_snapshots, lidar, dt=0.1):
    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_env(ax, env, show_names=False)
    ax.set_title('FrostBot — Pizza Fabrikası Simülasyonu', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    true_arr = np.array(true_path)
    ekf_arr  = np.array(ekf_path)

    true_line, = ax.plot([], [], 'g-',  lw=1.5, label='Gerçek Yol',   zorder=5)
    ekf_line,  = ax.plot([], [], 'b--', lw=1.0, label='EKF Tahmini',  zorder=4)
    robot_dot, = ax.plot([], [], 'go',  ms=10,  zorder=7)
    ekf_dot,   = ax.plot([], [], 'b^',  ms=7,   zorder=6, label='EKF Konum')
    lidar_sc   = ax.scatter([], [], s=2, c='cyan', alpha=0.45, zorder=3, label='LiDAR')
    time_txt   = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                         va='top', fontsize=9, color='#2c3e50',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    ax.legend(loc='upper right', fontsize=8)

    snap_dict = {s: nd for s, _td, nd in lidar_snapshots}
    N         = len(true_path)
    stride    = max(1, N // 180)
    frames    = list(range(0, N, stride))

    def _update(fi):
        step = frames[fi]
        true_line.set_data(true_arr[:step+1, 0], true_arr[:step+1, 1])
        ekf_line.set_data(ekf_arr[:step+1, 0],   ekf_arr[:step+1, 1])
        robot_dot.set_data([true_arr[step, 0]], [true_arr[step, 1]])
        ekf_dot.set_data([ekf_arr[step, 0]],   [ekf_arr[step, 1]])

        nearest = min(snap_dict.keys(), key=lambda s: abs(s - step))
        nd      = snap_dict[nearest]
        px, py  = lidar.to_cartesian(true_arr[step], nd)
        lidar_sc.set_offsets(np.c_[px, py])

        time_txt.set_text(f't = {step * dt:.1f} s')
        return true_line, ekf_line, robot_dot, ekf_dot, lidar_sc, time_txt

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=50, blit=True)

    path = os.path.join(OUTPUT_DIR, 'animation.gif')
    anim.save(path, writer=animation.PillowWriter(fps=20))
    print(f"  Saved: {path}")
    plt.close(fig)
