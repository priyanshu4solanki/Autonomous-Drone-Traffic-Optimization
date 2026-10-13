# Autonomous Drone Traffic Optimization

A city-grid simulation for coordinating a fleet of autonomous delivery drones. The project uses **Multi-Agent Proximal Policy Optimization (MAPPO)** and **Centralized Training with Decentralized Execution (CTDE)** to learn delivery behavior, with a convolutional neural network (CNN) that processes a local map around each drone.

This is a research and teaching simulation. It does not connect to or control real drone hardware.

## Quick Start

```powershell
cd Autonomous-Drone-Traffic-Optimization
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The interactive demo uses a CNN checkpoint at `checkpoints/cnn/mappo_best.pt`. CNN training checkpoints are generated locally and are not included in this repository, so create one before launching the default CNN demo:

```powershell
python train_mappo.py
python demo.py
```

Training uses 1,000 episodes by default and may take time. To try the interactive demo without training first, use the included older MLP checkpoint instead:

```powershell
python evaluate_mappo.py --interactive --model checkpoints/mappo_best.pt
```

The MLP option runs the simulator and renderer, but does not use the CNN map features.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Core Approach](#core-approach)
4. [Simulation Environment](#️-simulation-environment)
5. [Drone Tasks](#-drone-tasks)
6. [Action Space](#-action-space)
7. [Local Observation](#-local-observation)
8. [Global State](#-global-state)
9. [Reward System](#-reward-system)
10. [Safety Mechanisms](#️-safety-mechanisms)
11. [MAPPO Architecture](#-mappo-architecture)
12. [PPO + GAE](#-ppo--gae)
13. [Project Structure](#-project-structure)
14. [Main Components](#-main-components)
15. [Testing](#-testing)
16. [Installation](#️-installation)
17. [Training](#️-training)
18. [Evaluation](#-evaluation)
19. [Current Configuration](#-current-configuration)
20. [Future Work](#-future-work)
21. [Technologies](#️-technologies)
22. [Project Status](#-project-status)

---

## Project Overview

The project simulates a fleet of delivery drones operating at the same time in a city. Training and non-interactive evaluation use five drones by default; the interactive demo asks how many drones to launch.

Each drone must navigate from an assigned warehouse to a delivery point while it:

- Reaches its assigned delivery point
- Avoids buildings, restricted zones, and other drones
- Manages battery use and charging
- Completes its task efficiently

The policy is learned through reinforcement learning rather than by manually scripting every drone's actions. An optional A* route-guidance mode is also available for comparison and demonstrations; results from that mode are not CNN-only results.

---

## Core Approach

The system follows **Centralized Training and Decentralized Execution (CTDE)**.

The environment exposes a Gymnasium-compatible joint action/observation API. Each `step()` returns a scalar team reward for Gymnasium; the per-drone rewards used by MAPPO are provided as `info["agent_rewards"]`.

```
                         TRAINING
                            │
                            ▼
                   Global Environment
                            │
                            ▼
                   ┌────────────────┐
                   │ Central Critic │
                   └────────┬───────┘
                            │
                            ▼
                    Value Estimates
                            │
                            ▼
                   MAPPO / PPO Update
                            │
                            ▼
                      Shared Actor
                            │
                            ▼
                     Trained Policy
                            │
                            ▼
                         EXECUTION
                            │
              ┌─────────────┼─────────────┐
              ▼              ▼              ▼
          Drone 1        Drone 2        Drone N
              │              │              │
              ▼              ▼              ▼
        Local Obs + Map Local Obs + Map Local Obs + Map
              │              │              │
              ▼              ▼              ▼
           Action         Action         Action
```

**During Training** — the Central Critic has access to global information about all drones.

**During Execution** — each drone uses only its own **local observation and local map crop** to make decisions. The global state is not sent to the actors.

This allows the trained system to operate without requiring every drone to continuously access the complete global environment state.

---

## 🏙️ Simulation Environment

A custom **Gymnasium** environment represents a small city containing:

| Element | Description |
|---|---|
| 🏢 Buildings | Static obstacles that block movement |
| 🚫 No-Fly Zones | Restricted airspace, entry is penalized |
| 📦 Warehouses | Delivery task sources |
| 📍 Delivery Points | Delivery task destinations |
| ⚡ Charging Stations | Restore battery when visited |
| 🚁 Drones | 5 by default; configurable, or prompted in the interactive demo |

The environment supports seeded random generation, allowing different scenarios to be created while maintaining reproducibility.

---

## 🚁 Drone Tasks

Each drone receives a delivery task:

```
Warehouse → Delivery Point
```

During an episode, the drone must navigate from its source to its assigned destination while handling environmental constraints and other drones.

---

## 🎮 Action Space

Each drone can choose one of six actions:

| Action | Description |
|:---:|---|
| `0` | Move Up |
| `1` | Move Down |
| `2` | Move Left |
| `3` | Move Right |
| `4` | Wait |
| `5` | Recharge |

All drones select actions simultaneously.

---

## 👀 Local Observation

Each drone receives an **18-dimensional local observation** containing information such as:

- Relative destination
- Battery level
- Nearby obstacles
- Nearby drones
- Nearest-drone information
- Local traffic density
- Charging station availability
- Delivery status

```
Number of drones × 18 local features
```

The same Actor network is shared across all drones. It also receives a **6-channel, 9 × 9 map crop** around each drone (with the default observation radius of 4 cells). The channels mark buildings, no-fly zones, charging stations, other drones, the observing drone, and its destination. Cells outside the city are marked as blocked.

The map is built from simulator state; it is not a camera image, and the CNN does not perform object detection. During MAPPO training, the CNN learns spatial patterns from these map channels.

---

## 🌐 Global State

During training, the Central Critic receives a global representation of the environment.

For 5 drones (the default):

```
5 × 7 drone features + 3 global features
= 38-dimensional global state
```

The Critic produces:

```
5 value estimates — one for each drone by default
```

---

## 🧮 Reward System

The reward function encourages successful, safe, and efficient deliveries.

| Event | Reward |
|---|---:|
| Successful delivery | +100 |
| Attempted conflicting move | -80 |
| Illegal action | -2 |
| Battery failure | -40 |
| Progress toward destination | +2 |
| Waiting | -0.10 |
| Each step | -0.05 |

The objective is therefore not simply to reach the destination, but to do so **safely and efficiently**.

---

## 🛡️ Safety Mechanisms

The environment includes several safety mechanisms.

**Action Masking** — invalid actions are prevented from being selected, including:

- Moving outside the city
- Entering buildings
- Entering no-fly zones
- Moving into a cell currently occupied by another drone
- Moving without sufficient battery
- Recharging away from a charging station

**Joint-action conflict resolution** — after all drones propose moves, the environment arbitrates multiple drones targeting the same free cell (one proceeds, the others wait), and cancels moves involved in:

- A drone entering a cell occupied by a delivered or inactive drone
- Two drones swapping positions in the same timestep (normally prevented by occupancy masks)

Canceled moves do not consume movement battery. The evaluation's `Conflicts` metric counts contested drone-pair moves, not physical collisions; the renderer shows the same count.

Task generation assigns distinct start cells and distinct destination cells, and keeps starts separate from destinations. If the city does not have enough free cells for a fleet, reset fails with a clear error instead of silently creating overlapping tasks.
---

## 🤖 MAPPO Architecture

The project implements MAPPO from scratch using PyTorch.

**Shared decentralized Actor**
```
Local map: 6 × 9 × 9 → convolutional layers → 128 map features
18 observation values + 128 map features → 128 → 128 → 6
```
The CNN encodes nearby map layout. Its features are combined with the drone's 18-value local observation, and the actor produces a probability distribution over the six actions. Legal-action masking is still applied.

**Central Critic**
```
38 → 128 → 128 → 5 (default 5-drone configuration)
```
The Critic receives global state during training and estimates each drone's value. During execution, only the shared actor's local inputs are needed.

---

## 📈 PPO + GAE

The MAPPO implementation uses:

- PPO clipped objective
- Advantage normalization
- Entropy regularization
- Gradient clipping
- Mini-batch updates
- Generalized Advantage Estimation (GAE)

GAE uses rewards and Critic value estimates to calculate:

```
Advantages → Actor update
Returns    → Critic update
```

This helps provide more stable policy learning.

---

## 📂 Project Structure

```
Autonomous-Drone-Traffic-Optimization/
│
├── environment/
│   ├── __init__.py
│   ├── drone_env.py
│   ├── drone.py
│   ├── city.py
│   ├── obstacles.py
│   ├── task_generator.py
│   ├── observations.py
│   ├── movement.py
│   ├── collisions.py
│   ├── rewards.py
│   └── renderer.py
│
├── mappo/
│   ├── __init__.py
│   ├── agent.py
│   ├── route_guidance.py
│   ├── networks.py
│   └── rollout.py
│
├── checkpoints/
│   ├── mappo.pt
│   └── mappo_best.pt
│
├── train_mappo.py
├── evaluate_mappo.py
├── demo.py
├── test_random_city.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Main Components

| File | Purpose |
|---|---|
| `drone_env.py` | Main Gymnasium environment |
| `city.py` | City and obstacle generation |
| `drone.py` | Drone state and tracking |
| `obstacles.py` | Building and no-fly zone structures |
| `task_generator.py` | Delivery task generation |
| `observations.py` | Local observations, CNN map crops, and global state |
| `movement.py` | Movement, battery and action masking |
| `collisions.py` | Multi-drone collision detection |
| `rewards.py` | Reward calculation |
| `renderer.py` | City view, selected drone's local map, and action probabilities |
| `networks.py` | CNN Actor and Central Critic |
| `rollout.py` | GAE and return calculation |
| `agent.py` | MAPPO training logic |
| `train_mappo.py` | Training pipeline |
| `evaluate_mappo.py` | Interactive demo and model evaluation |
| `demo.py` | Starts the interactive delivery demo |
| `test_random_city.py` | Random city generation testing |

---

## 🧪 Testing

The project includes environment and CNN map-input tests, plus `test_random_city.py` to inspect random city generation.

Run:

```bash
python -m pytest
python test_random_city.py
```

The test helps verify that the environment can generate different city configurations using different random seeds. This supports the use of randomized environments during training and testing.

---

## 🚀 Installation

```bash
git clone https://github.com/AadityaSaxena21/Autonomous-Drone-Traffic-Optimization.git
cd Autonomous-Drone-Traffic-Optimization
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

Train the CNN-based MAPPO actor from scratch:

```bash
python train_mappo.py
```

By default, the new CNN model is saved to `checkpoints/cnn/`, leaving the included legacy MLP checkpoints in `checkpoints/` untouched. Pass `--policy-type mlp` to train an MLP baseline in `checkpoints/mlp/`, or use `--checkpoint-dir` to choose another output directory.

The training process:

```
Environment
    ↓
Collect Experience
    ↓
Calculate GAE
    ↓
MAPPO Update
    ↓
Update Actor + Critic
    ↓
Repeat
    ↓
Save Checkpoint
```

The trained models are saved in `checkpoints/cnn/` (generated checkpoints are ignored by Git). The included checkpoints in `checkpoints/` were made with the earlier MLP actor and remain loadable for evaluation; run training to create a CNN checkpoint. For a quick smoke test without overwriting the main CNN checkpoint, use `python train_mappo.py --episodes 2 --max-steps 20 --checkpoint-dir checkpoints/smoke`.

---

## 📊 Evaluation

For a guided interactive presentation, run:

```powershell
python demo.py
```

The demo asks how many drones to launch, the step limit, city seed, playback speed, and whether to enable A* route guidance. It then shows numbered warehouse and delivery-point menus so you can assign a mission to each drone, or press Enter to use generated missions.

The window is paced for a live review. It shows drone positions and destinations, battery, delivery status, a map legend, the selected drone's local map, and action probabilities. Controls: **Space** pause/resume, **N** advance one step, **+/-** adjust speed, **1–9** or **Tab** inspect a drone, and **Esc** exit. The results screen remains until you press Enter or Esc.

Evaluate a trained model using:

```bash
python evaluate_mappo.py --model checkpoints/cnn/mappo_best.pt --episodes 10 --render
```

The Pygame side panel shows the selected drone's local map and action probabilities; the selected action is highlighted. Press `1`–`9` to inspect another drone. For a CNN checkpoint, this is the map input the CNN actually receives. The bundled `checkpoints/mappo_best.pt` is an older MLP model; it still loads, but it does not use the CNN map input. Evaluation uses deterministic actions only when `--deterministic` is passed.

For much stronger navigation using the simulator's known obstacle map, enable the optional A* route guidance:

```bash
python evaluate_mappo.py --model checkpoints/cnn/mappo_best.pt --episodes 30 --render --deterministic --route-guidance
```

This mode is an **A* route-planning baseline**, not evidence that the CNN alone achieves the reported delivery rate. The planner reads the simulator's full map and gives the CNN policy's action distribution only a fallback role when the planned action is unavailable. Compare against plain CNN by omitting `--route-guidance`.

The evaluation tracks:

- Delivery rate
- Total deliveries
- Collisions
- No-fly violations
- Battery remaining
- Episode reward
- Steps per episode

A model trained with this project is for simulation experiments, not a validated flight controller. This project has no sensor ingestion, localization, real-time flight integration, or hardware safety certification. Do not connect it to a real drone.

---

## 📌 Current Configuration

| Setting | Value |
|---|---|
| Number of drones | 5 (default; configurable) |
| Local observation | 18 |
| Local CNN map | 6 channels × 9 × 9 cells |
| Global state | 38 (default; changes with drone count) |
| Actions per drone | 6 |
| CNN Actor | Map features + 18 observation values → 6 actions |
| Critic architecture | 38 → 128 → 128 → 5 (default) |
| Algorithm | MAPPO |
| Training paradigm | CTDE |
| Environment | Gymnasium |
| Deep Learning | PyTorch |

---

## 🔬 Future Work

Potential extensions include:

- 🌐 Graph Neural Networks for drone interaction
- 📡 Communication-aware multi-agent policies
- 🗺️ Larger and more realistic city maps
- 🌦️ Weather and wind simulation
- 🔋 More realistic battery models
- 🚁 Continuous 2D/3D drone movement
- 📦 Dynamic task assignment
- 📊 Baseline comparison with other MARL algorithms
- 📈 Large-scale fleet experiments

---

## 🛠️ Technologies

- Python
- PyTorch
- Gymnasium
- NumPy
- Pygame
- MAPPO
- PPO
- GAE
- Multi-Agent Reinforcement Learning

---

## 📜 Project Status

🚧 **Research / Development Project**

The current implementation provides the complete foundation for a multi-drone MARL system, including environment simulation, local observations, centralized training, decentralized execution, collision handling, battery management, MAPPO training, randomized city testing, and evaluation.

The next stage focuses on improving scalability, generalization, collision avoidance, and experimental evaluation.

---

## ⭐ Key Takeaway

> **Train globally. Execute locally. Coordinate intelligently.**

This project explores how multiple autonomous drones can learn safe and efficient delivery behavior through **Multi-Agent Reinforcement Learning** rather than relying entirely on manually designed navigation rules.
