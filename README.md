# MAE598 — Multi-Robot Systems: Swarm Firefighting & Rescue Simulations

> **Course:** MAE598 – Multi-Robot Systems  
> **Language:** MATLAB  
> **Topics:** Swarm intelligence, task allocation, sensor fusion, collision avoidance

---

## Overview

This repository contains MATLAB simulations developed for the MAE598 Multi-Robot Systems course project. Two emergency-response scenarios are implemented and progressively refined:

1. **Swarm Firefighting** — a swarm of robots detects and suppresses multiple fire hotspots on a live heatmap, using greedy task assignment and proximity-based fire suppression.
2. **Swarm Rescue** — robots explore an unknown environment, detect a downed human using noisy sensors, and fuse their observations via Weighted Least Squares (WLS) to localize and converge on the target.

Each scenario is built up through several script variants that introduce added complexity such as cone-of-vision sensing, refill mechanics, multi-target rescue, loitering behavior, and sequential tasking.

---

## Repository Structure

```
MAE598_Multi-Robot-Systems/
│
├── swarm_firefighter.m            # Base firefighting simulation
├── swarm_firefighter_cone.m       # Firefighting with cone-of-vision sensing
├── swarm_firefighter_cone_refill.m # Cone sensing + agent refill/recharge mechanic
│
├── swarm_rescue.m                 # Base rescue simulation (single target, WLS fusion)
├── swarm_rescue_multi.m           # Rescue with multiple targets
├── swarm_rescue_seq.m             # Sequential rescue task execution
├── swarm_rescue_seq_loiter.m      # Sequential rescue with loiter-on-arrival behavior
│
├── MRS_Final_Report.pdf           # Full project report
├── 1728016449.7729878.MOV         # Simulation demo video
│
└── *.bib / *.ris / *.txt          # References and citation files
```

---

## Scenarios

### 🔥 Swarm Firefighting

**Files:** `swarm_firefighter.m`, `swarm_firefighter_cone.m`, `swarm_firefighter_cone_refill.m`

A fleet of **N = 10 robots** operates over a 60 × 40 m map containing **M = 5 randomly placed fire hotspots**, each modeled as a 2D Gaussian heat source. The simulation runs at 15 Hz (dt = 0.15 s) for up to 300 simulated seconds.

**Key behaviors:**
- **Heatmap-driven detection** — the environment is represented as a pixel-resolution heat field rebuilt each timestep from active hotspot Gaussians.
- **Greedy task assignment** — every `reassign_period` seconds, active fire peaks are identified via regional maximum detection (`imregionalmax`). Robots are assigned to fires by nearest-distance greedy matching.
- **Fire suppression** — each robot within `ext_radius = 2.0 m` of a hotspot reduces its amplitude multiplicatively (`decay_rate^(dt × closeCnt)`); fires with amplitude < 0.1 are marked extinguished.
- **Separation/collision avoidance** — a repulsive potential kicks in when two robots come within `1.2 × bbox` of each other, preventing collisions.
- **Search behavior** — unassigned robots perform a correlated random walk across the map.
- **Cone sensing variant** — `swarm_firefighter_cone.m` restricts each robot's heat sensing to a forward-facing cone, making detection more realistic.
- **Refill variant** — `swarm_firefighter_cone_refill.m` adds a finite suppressant capacity per robot, requiring agents to return to a refill station before continuing.

**Visualization:** Live animated heatmap (color = intensity), cyan robot markers with bounding boxes, white circles for active fires, green stars for extinguished fires.

---

### 🧍 Swarm Rescue

**Files:** `swarm_rescue.m`, `swarm_rescue_multi.m`, `swarm_rescue_seq.m`, `swarm_rescue_seq_loiter.m`

**N = 8 robots** search a 50 × 50 m map for one or more downed humans. Robots begin in random positions and switch from random-walk exploration to directed response upon detection.

**Key behaviors:**
- **Detection** — any robot within `detectRadius = 12 m` of a human generates a noisy position measurement (Gaussian noise, σ = 0.8 m).
- **Sensor fusion (WLS)** — all detections collected in a timestep are fused using Weighted Least Squares to produce a best estimate `x̂` and covariance `P`. The simulation terminates when `trace(P) < τ = 0.25 m²`, indicating sufficient localization confidence.
- **Mode switching** — upon any detection, all robots switch from `"Search"` to `"Respond"` mode and navigate toward the fused estimate.
- **Multi-target variant** — `swarm_rescue_multi.m` handles multiple simultaneous targets with separate fusion channels.
- **Sequential variant** — `swarm_rescue_seq.m` chains rescue tasks, sending robots to the next target only after the current one is localized.
- **Loiter variant** — `swarm_rescue_seq_loiter.m` adds loitering-on-arrival behavior so robots orbit a confirmed target while the rest of the swarm continues searching.

**Visualization:** Robot positions (blue), ground-truth human location (red ×), fused estimate (black +), 95% confidence ellipse drawn from the fused covariance.

---

## Getting Started

### Prerequisites

- MATLAB R2021a or later (R2022b+ recommended)
- Image Processing Toolbox (required for `imregionalmax` in the firefighting scripts)

### Running a Simulation

Open MATLAB, navigate to the repository folder, and call any simulation function:

```matlab
% Firefighting (base)
swarm_firefighter()

% Firefighting with cone sensing
swarm_firefighter_cone()

% Rescue (base)
swarm_rescue()

% Rescue with multiple targets
swarm_rescue_multi()
```

Each script is self-contained — no external data files or additional setup are required.

### Key Parameters to Tune

| Parameter | Location | Description |
|---|---|---|
| `N` | All scripts | Number of robots |
| `M` | Firefighting | Number of fire hotspots |
| `decay_rate` | Firefighting | Fire suppression rate per second |
| `ext_radius` | Firefighting | Suppression influence radius (m) |
| `reassign_period` | Firefighting | Seconds between task reassignments |
| `detectRadius` | Rescue | Sensor detection range (m) |
| `Sigma_z` | Rescue | Measurement noise covariance |
| `tau_cov` | Rescue | Localization confidence threshold (trace of P) |
| `dt` | All scripts | Simulation timestep (s) |

---

## Algorithms & Methods

| Concept | Implementation |
|---|---|
| Task allocation | Greedy nearest-robot assignment, periodic reassignment |
| Fire detection | Regional maximum finding on normalized heatmap (`imregionalmax`) |
| Collision avoidance | Repulsive potential field between neighboring robots |
| Sensor fusion | Weighted Least Squares (WLS) over multi-robot detections |
| Localization confidence | Covariance trace threshold (`trace(P) < τ`) |
| Motion model | Euler integration with additive Gaussian process noise |
| Exploration | Correlated random walk with periodic direction refresh |

---

## References

See `MRS_Final_Report.pdf` for full citations. Key references include the `.bib` and `.ris` files in the repository root (Elsevier articles `S0009250903004652` and `S0959152423002317`).

---

## Authors

Developed as a final project for **MAE598 – Multi-Robot Systems** at Arizona State University.
