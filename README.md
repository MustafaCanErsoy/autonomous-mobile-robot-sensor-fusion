# FrostBot — Autonomous Mobile Robot Sensor Fusion

**2D LiDAR-based autonomous navigation simulation with Extended Kalman Filter (EKF) sensor fusion, localization, and multi-algorithm path planning in a frozen pizza factory environment.**

> Bursa Teknik Üniversitesi — Bilgisayar Mühendisliği — Mekatronik Teknik Seçmeli  
> Öğrenci: Mustafa Can Ersoy

---

## Senaryo / Scenario

A differential-drive robot navigates a **50 × 50 m frozen pizza factory** autonomously, transporting pizzas from the production line to cold storage. The factory contains heavy machinery, conveyor belts, and structural pillars that block direct paths and emit electromagnetic interference, making GPS unusable. The robot relies solely on:

- **LiDAR** — 180-beam 2D scan, max range 12 m
- **IMU** — angular velocity measurement
- **Wheel encoder** — linear and angular velocity from wheel speeds

The robot must reach three waypoints in order:

| # | Waypoint | Coordinates |
|---|----------|-------------|
| 1 | Kalite Kontrol (Quality Control) | (14, 12) m |
| 2 | Ambalaj (Packaging) | (28, 28) m |
| 3 | Soğuk Depo (Cold Storage) | (45, 45) m |

Sensor noise is **3× higher** near heavy machinery (electromagnetic interference zones).

---

## Features

- **Non-holonomic differential-drive** kinematic model
- **3-sensor simulation** with zone-dependent Gaussian noise (LiDAR, IMU, encoder)
- **Extended Kalman Filter (EKF)** — encoder prediction + IMU update, with Jacobian linearisation of the nonlinear motion model
- **Dead reckoning** baseline for localization error comparison
- **Two navigation algorithms**: Potential Field vs Bug2 (with stuck-escape mechanism)
- **Quantitative error analysis**: RMSE and MAE for EKF vs Dead Reckoning
- **Animated GIF** simulation output
- Fully modular Python codebase — each component is independently testable

---

## Installation

```bash
git clone https://github.com/MustafaCanErsoy/autonomous-mobile-robot-sensor-fusion.git
cd autonomous-mobile-robot-sensor-fusion
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, numpy, matplotlib, scipy, pillow

---

## Usage

```bash
python main.py
```

All outputs are saved to the `outputs/` directory:

| File | Description |
|------|-------------|
| `01_environment.png` | Factory map — obstacles, noise zones, waypoints |
| `02_path_comparison.png` | Potential Field vs Bug2 path overlay |
| `03_lidar_data.png` | Raw vs noisy LiDAR scan point cloud |
| `04_localization.png` | Ground truth vs EKF vs Dead Reckoning (2D + time-series) |
| `05_error_analysis.png` | Position error over time, RMSE and MAE |
| `animation.gif` | Real-time animated simulation with LiDAR rays |

---

## Project Structure

```
├── main.py               # Entry point — runs both simulations, saves all outputs
├── environment.py        # Factory map: 15 obstacles, vectorised ray casting
├── robot.py              # Differential-drive kinematic model
├── sensors/
│   ├── lidar.py          # 2D LiDAR with noise and obstacle clustering
│   ├── imu.py            # Angular velocity sensor
│   └── encoder.py        # Wheel encoder (v, omega measurement)
├── fusion/
│   └── ekf.py            # Extended Kalman Filter (predict + update)
├── localization.py       # Dead reckoning + RMSE/MAE error metrics
├── navigation.py         # PotentialFieldNav and Bug2Nav
├── visualization.py      # All plots and animation
└── requirements.txt
```

---

## Methods

### Sensor Fusion — Extended Kalman Filter

State vector: **[x, y, θ]**

**Prediction step** (wheel encoder as control input):

```
x'  = x + v·cos(θ)·dt
y'  = y + v·sin(θ)·dt
θ'  = θ + ω·dt
P'  = F·P·Fᵀ + Q
```

where F is the Jacobian of the nonlinear motion model. EKF differs from standard KF here — the nonlinear kinematics require linearisation at each step.

**Update step** (IMU angular velocity):

The IMU provides a virtual theta observation: `z = θ_prev + ω_imu·dt`. Innovation `z − Hx` captures the discrepancy between IMU and encoder angular rates, correcting theta drift.

### Navigation

**Potential Field:** Attractive force toward goal + repulsive forces from LiDAR obstacle readings within 2.5 m. Includes stuck-detection with random escape perturbation (triggered if robot moves < 12 cm in 30 steps).

**Bug2:** State machine with GO_TO_GOAL and FOLLOW_WALL modes. When an obstacle is detected ahead (< 1.3 m), the robot follows the right-side wall until it re-crosses the M-line (start–goal line) closer to the goal.

### Dead Reckoning (baseline)

Pure encoder integration — no fusion. Accumulates drift over time, used as the lower-bound benchmark for localization quality.

---

## Results

### Navigation

| Algorithm | Waypoints Reached | Simulation Time |
|-----------|------------------|-----------------|
| Potential Field | 3 / 3 | 83.0 s |
| Bug2 | 2 / 3 | 400.0 s (timeout at WP3) |

Potential Field outperforms Bug2 in this environment. Bug2's wall-following mode gets trapped near WP3 due to the tight cluster of obstacles (cold storage area + conveyor belts), a known limitation of the algorithm in complex multi-obstacle layouts.

### Localization (Potential Field run, seed=42)

| Metric | EKF | Dead Reckoning |
|--------|-----|----------------|
| RMSE (m) | **6.631** | 13.750 |
| MAE (m) | **5.298** | 10.992 |

**EKF reduces position error by ~51.8% vs Dead Reckoning** over an 83-second, 60-meter traverse in a GPS-denied environment. Error magnitudes reflect the absence of absolute position correction (no GPS, no landmark matching) — the EKF drifts, but significantly less than pure odometry.

---

## References

[1] V. Ušinskis, M. Nowicki, A. Dzedzickis and V. Bučinskas, "Sensor-fusion based navigation for autonomous mobile robot," *Sensors*, vol. 25, no. 4, article 1248, 2025. doi: 10.3390/s25041248

[2] Y. Ou, Y. Cai, Y. Sun and T. Qin, "Autonomous navigation by mobile robot with sensor fusion based on deep reinforcement learning," *Sensors*, vol. 24, no. 12, article 3895, 2024. doi: 10.3390/s24123895

[3] B. Zhang and C. Li, "The optimization and application research of the RRT-APF-based path planning algorithm," *Electronics*, vol. 13, no. 24, article 4963, 2024. doi: 10.3390/electronics13244963

---

## AI Usage Declaration

| Tool | Version | Sections |
|------|---------|----------|
| Claude Code (Anthropic) | claude-sonnet-4-6 | System architecture design, EKF formulation, navigation algorithm implementation, visualization code, README |

**Student contributions:** Scenario concept and design (frozen pizza factory, waypoint layout), algorithm selection and parameter tuning decisions, result interpretation and analysis, project direction throughout development.

AI tools were used as a collaborative coding assistant. All code was reviewed, executed, and validated by the student.
