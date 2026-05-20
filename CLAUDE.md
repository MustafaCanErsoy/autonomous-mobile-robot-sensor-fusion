# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic Python simulation project for a Bursa Teknik Üniversitesi 4th-year mechatronics elective course (10% of grade). The topic is **sensor fusion and localization in a 2D LiDAR-based autonomous mobile robot simulation**.

The professor explicitly encourages extra effort beyond the base requirements — do more than just satisfy the checklist.

## Running the Code

```bash
python main.py
```

Runs both simulations (Potential Field + Bug2), generates all plots, and saves them to `outputs/`. The directory is wiped clean before every run.

## Required Python Libraries

```bash
pip install numpy matplotlib pillow
```

## Architecture

The codebase is organized as independent modules wired together in `main.py`:

| Module | Responsibility |
|---|---|
| `environment.py` | 2D map: obstacles (≥10), start/goal points, boundaries |
| `robot.py` | Non-holonomic differential-drive kinematics |
| `sensors/lidar.py` | 2D LiDAR scan simulation with noise, distance thresholding, obstacle clustering |
| `sensors/imu.py` | IMU simulation with noise |
| `sensors/encoder.py` | Wheel encoder / odometry simulation |
| `fusion/ekf.py` | Extended Kalman Filter combining ≥2 sensor streams (mandatory) |
| `localization.py` | Dead reckoning + fused estimation; real vs estimated path comparison |
| `navigation.py` | Obstacle avoidance + goal reaching (Potential Field, VFH, or reactive) |
| `visualization.py` | All required plots |

## Hard Technical Constraints

- **Kalman Filter is mandatory** for sensor fusion — do not substitute with a simpler weighted average.
- **Non-holonomic robot model** (differential drive) — not a holonomic point mass.
- **At least 2 sensors** must be fused (from: LiDAR, IMU, wheel encoder).
- **Minimum 10 obstacles** in the 2D environment.
- Sensor noise must be explicitly modeled (not zero).

## Required Visualizations (all must be generated)

1. **Environment map** — top-down 2D view with obstacles, start (green), goal (red/magenta), boundaries.
2. **Path comparison** — planned vs actual path, same graph, with start/goal/obstacles shown.
3. **Sensor data** — raw vs filtered LiDAR scan points (separate colors or subplots).
4. **Localization results** — real path vs estimated path (2D or time-series x(t), y(t), θ(t)).
5. **Error analysis** — position error over time with RMSE or MAE metric labeled on graph.

Every plot must have: title, axis labels with units, legend.

## Deliverables Checklist

- [ ] Fully working Python simulation code
- [ ] GitHub repository
- [ ] README.md (installation + usage instructions)
- [ ] PDF report: intro + scenario, methods, results + graphs, error analysis, references (IEEE/APA), AI usage declaration
- [ ] All 5 visualizations above saved as image files

## AI Usage Declaration (required in report)

The final PDF report must include an AI usage declaration section listing which AI tools were used (e.g., Claude Sonnet 4.6), which sections they assisted with, and what the student contributed independently.
