"""
FrostBot — Pizza Fabrikası Otonom Navigasyon Simülasyonu
=========================================================
2D LiDAR + IMU + Enkoder sensör füzyonu (Extended Kalman Filter).
Potential Field ve Bug2 navigasyon algoritmaları karşılaştırması.

Kullanım:
    python main.py
"""

import os
import shutil
import numpy as np

from environment import Environment
from robot import DifferentialDriveRobot
from sensors.lidar import LiDAR
from sensors.imu import IMU
from sensors.encoder import WheelEncoder
from fusion.ekf import ExtendedKalmanFilter
from localization import DeadReckoning
from navigation import PotentialFieldNav, Bug2Nav
from visualization import (
    plot_environment, plot_paths, plot_lidar,
    plot_localization, plot_errors, create_animation,
    plot_lidar_heatmap,
)

DT        = 0.1    # simulation timestep (s)
MAX_STEPS = 4000   # ~400 s max simulation time
GOAL_TOL  = 1.5    # waypoint reached threshold (m)


def _path_length(true_path, up_to_step):
    """Total Euclidean distance travelled along true_path up to given step."""
    arr = np.array(true_path[:up_to_step + 1])
    return float(np.sum(np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)))


# ------------------------------------------------------------------ #

def run_simulation(navigator_type='potential_field', seed=42, verbose=True):
    np.random.seed(seed)

    env    = Environment()
    robot  = DifferentialDriveRobot(env.start[0], env.start[1], theta=np.pi / 4)
    lidar  = LiDAR(n_beams=180, max_range=12.0, noise_std=0.05)
    imu    = IMU(omega_noise_std=0.015)
    enc    = WheelEncoder(noise_std=0.03)

    ekf = ExtendedKalmanFilter(
        initial_state=[env.start[0], env.start[1], np.pi / 4],
        P0=np.diag([0.1, 0.1, 0.05]),
        Q=np.diag([0.005, 0.005, 0.002]),
        R_imu=np.array([[0.003]]),
    )
    dr = DeadReckoning(env.start[0], env.start[1], np.pi / 4)

    if navigator_type == 'potential_field':
        nav = PotentialFieldNav()
    else:
        nav = Bug2Nav(env.start.copy())

    true_path        = [robot.state.copy()]
    ekf_path         = [ekf.state.copy()]
    dr_path          = [dr.state.copy()]
    lidar_snapshots  = []          # (step, true_dist, noisy_dist) every 15 steps
    cov_history      = []          # (step, xy_pos, 2x2_cov) every 30 steps
    waypoint_log     = []          # {name, step, time} when each WP is reached
    forklift_history = [(env.forklift.x, env.forklift.y)]  # (fx, fy) — aligned with true_path
    heatmap          = np.zeros((100, 100))   # LiDAR hit-point density
    wp_idx           = 0

    for step in range(MAX_STEPS):
        goal = env.waypoints[wp_idx]

        # Waypoint reached?
        if np.linalg.norm(robot.state[:2] - goal) < GOAL_TOL:
            waypoint_log.append({
                'name': env.waypoint_names[wp_idx],
                'step': step,
                'time': step * DT,
            })
            if verbose:
                print(f"  [{navigator_type}] WP{wp_idx + 1} '{env.waypoint_names[wp_idx]}'"
                      f" reached at t = {step * DT:.1f} s")
            wp_idx += 1
            if wp_idx >= len(env.waypoints):
                if verbose:
                    print(f"  [{navigator_type}] All waypoints reached! "
                          f"t = {step * DT:.1f} s  ({step} steps)")
                break
            if navigator_type == 'bug2':
                nav.reset(robot.state[:2].copy())
            goal = env.waypoints[wp_idx]

        # ---- Sensor readings ----
        true_dist, noisy_dist = lidar.scan(robot.state, env)
        omega_imu             = imu.measure(robot, env)
        v_enc, omega_enc      = enc.measure(robot, env)

        # ---- EKF: predict (encoder) + update (IMU) ----
        theta_prev = ekf.state[2]
        ekf.predict(v_enc, omega_enc, DT)
        ekf.update_theta(theta_prev + omega_imu * DT)

        # ---- Dead reckoning (encoder only) ----
        dr.step(v_enc, omega_enc, DT)

        # ---- Navigation command ----
        v_cmd, omega_cmd = nav.compute(robot.state, goal, noisy_dist, lidar.angles)

        # ---- Move robot (ground truth) ----
        robot.step(v_cmd, omega_cmd, DT, env)

        # ---- Advance dynamic obstacles ----
        env.step(DT)

        # ---- Store data ----
        true_path.append(robot.state.copy())
        ekf_path.append(ekf.state.copy())
        dr_path.append(dr.state.copy())
        forklift_history.append((env.forklift.x, env.forklift.y))

        # LiDAR heatmap — accumulate hit points (exclude max-range beams)
        mask = noisy_dist < lidar.max_range * 0.95
        if np.any(mask):
            beam_ang = robot.state[2] + lidar.angles[mask]
            px = robot.state[0] + noisy_dist[mask] * np.cos(beam_ang)
            py = robot.state[1] + noisy_dist[mask] * np.sin(beam_ang)
            H, _, _ = np.histogram2d(px, py, bins=100,
                                     range=[[0, env.WIDTH], [0, env.HEIGHT]])
            heatmap += H

        if step % 15 == 0:
            lidar_snapshots.append((step, true_dist.copy(), noisy_dist.copy()))
        if step % 30 == 0:
            cov_history.append((step, ekf.state[:2].copy(), ekf.get_xy_covariance()))

    return dict(
        true_path=true_path,
        ekf_path=ekf_path,
        dr_path=dr_path,
        lidar_snapshots=lidar_snapshots,
        cov_history=cov_history,
        waypoint_log=waypoint_log,
        forklift_history=forklift_history,
        heatmap=heatmap,
        env=env,
        lidar=lidar,
    )


# ------------------------------------------------------------------ #

def main():
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
    os.makedirs('outputs')

    print("=" * 55)
    print("  FrostBot — Pizza Fabrikası Simülasyonu")
    print("=" * 55)

    print("\n[1/2] Potential Field simülasyonu çalışıyor...")
    pf = run_simulation('potential_field', seed=42)

    print("\n[2/2] Bug2 simülasyonu çalışıyor...")
    b2 = run_simulation('bug2', seed=42)

    env   = pf['env']
    lidar = pf['lidar']

    print("\nGrafikler oluşturuluyor...")

    # 1 — Environment map
    plot_environment(env)

    # 2 — Path comparison
    plot_paths(env, pf['true_path'], b2['true_path'])

    # 3 — LiDAR data (mid-simulation snapshot)
    mid                          = len(pf['lidar_snapshots']) // 2
    snap_step, true_d, noisy_d   = pf['lidar_snapshots'][mid]
    lidar_state                  = pf['true_path'][snap_step]
    plot_lidar(lidar_state, true_d, noisy_d, lidar, env)

    # 4 — Localization
    plot_localization(env, pf['true_path'], pf['ekf_path'], pf['dr_path'],
                      cov_history=pf['cov_history'])

    # 5 — Error analysis
    rmse_ekf, rmse_dr, mae_ekf, mae_dr = plot_errors(
        pf['true_path'], pf['ekf_path'], pf['dr_path'])

    print(f"\n  {'Metrik':<12} {'EKF':>10} {'Dead Reckoning':>16}")
    print(f"  {'-'*40}")
    print(f"  {'RMSE (m)':<12} {rmse_ekf:>10.4f} {rmse_dr:>16.4f}")
    print(f"  {'MAE (m)':<12} {mae_ekf:>10.4f} {mae_dr:>16.4f}")
    improvement = (1 - rmse_ekf / rmse_dr) * 100 if rmse_dr > 0 else 0
    print(f"\n  EKF, Dead Reckoning'e göre %{improvement:.1f} daha iyi RMSE")

    # Waypoint karşılaştırma tablosu
    print(f"\n  {'Waypoint':<22} {'PF Sure':>9} {'PF Mesafe':>11} {'Bug2 Sure':>11} {'Bug2 Mesafe':>13}")
    print(f"  {'-'*68}")
    if len(pf['waypoint_log']) != len(b2['waypoint_log']):
        print(f"  [UYARI] Waypoint sayıları eşleşmiyor: "
              f"PF={len(pf['waypoint_log'])}, Bug2={len(b2['waypoint_log'])}")
    for pf_wp, b2_wp in zip(pf['waypoint_log'], b2['waypoint_log']):
        pf_dist  = _path_length(pf['true_path'], pf_wp['step'])
        b2_dist  = _path_length(b2['true_path'], b2_wp['step'])
        print(f"  {pf_wp['name']:<22} {pf_wp['time']:>8.1f}s {pf_dist:>10.1f}m"
              f" {b2_wp['time']:>10.1f}s {b2_dist:>12.1f}m")

    # Animasyon
    # 6 — LiDAR heatmap
    plot_lidar_heatmap(env, pf['heatmap'])

    print("\nAnimasyon oluşturuluyor (bu birkaç dakika sürebilir)...")
    create_animation(env, pf['true_path'], pf['ekf_path'],
                     pf['lidar_snapshots'], lidar, pf['forklift_history'])

    print("\n" + "=" * 55)
    print("  Tüm çıktılar 'outputs/' klasörüne kaydedildi.")
    print("=" * 55)


if __name__ == '__main__':
    main()
