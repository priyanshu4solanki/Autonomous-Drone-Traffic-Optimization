# 🚁 Autonomous Drone Traffic Optimization

A **Multi-Agent Reinforcement Learning (MARL)** system that trains autonomous drones to perform delivery tasks while avoiding obstacles, collisions, no-fly zones, and battery failures.

The project uses **MAPPO (Multi-Agent Proximal Policy Optimization)** with **Centralized Training and Decentralized Execution (CTDE)**.

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Core Approach](#-core-approach)
3. [Simulation Environment](#️-simulation-environment)
4. [Drone Tasks](#-drone-tasks)
5. [Action Space](#-action-space)
6. [Local Observation](#-local-observation)
7. [Global State](#-global-state)
8. [Reward System](#-reward-system)
9. [Safety Mechanisms](#️-safety-mechanisms)
10. [MAPPO Architecture](#-mappo-architecture)
11. [PPO + GAE](#-ppo--gae)
12. [Project Structure](#-project-structure)
13. [Main Components](#-main-components)
14. [Testing](#-testing)
15. [Installation](#️-installation)
16. [Training](#️-training)
17. [Evaluation](#-evaluation)
18. [Current Configuration](#-current-configuration)
19. [Future Work](#-future-work)
20. [Technologies](#️-technologies)
21. [Project Status](#-project-status)

---

## 🎯 Project Overview

The goal is to coordinate **10 autonomous delivery drones** operating simultaneously in a simulated city.

Each drone must learn to:

- 📦 Reach its assigned delivery point
- 🏙️ Avoid buildings and no-fly zones
- 🚁 Avoid collisions with other drones
- 🔋 Manage battery usage
- ⚡ Use charging stations when required
- 🛣️ Find efficient routes
- 🤝 Learn cooperative multi-drone behavior

Instead of manually defining routes, the drones learn their behavior through **reinforcement learning**.

---

## 🧠 Core Approach

The system follows **Centralized Training and Decentralized Execution (CTDE)**.

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
          Drone 1        Drone 2        Drone 10
              │              │              │
              ▼              ▼              ▼
          Local Obs      Local Obs      Local Obs
              │              │              │
              ▼              ▼              ▼
           Action         Action         Action
```

**During Training** — the Central Critic has access to global information about all drones.

**During Execution** — each drone uses only its own **local observation** to make decisions.

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
| 🚁 10 Drones | Agents operating simultaneously |

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

All 10 drones select actions simultaneously.

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
10 drones × 18 local features
```

The same Actor network is shared across all drones.

---

## 🌐 Global State

During training, the Central Critic receives a global representation of the environment.

For 10 drones:

```
10 × 7 drone features + 3 global features
= 73-dimensional global state
```

The Critic produces:

```
10 value estimates — one for each drone
```

---

## 🧮 Reward System

The reward function encourages successful, safe, and efficient deliveries.

| Event | Reward |
|---|---:|
| Successful delivery | +100 |
| Collision | -80 |
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
- Moving without sufficient battery
- Recharging away from a charging station

**Collision Detection** — the system detects:

- Two drones occupying the same cell
- Two drones swapping positions in the same timestep

---

## 🤖 MAPPO Architecture

The project implements MAPPO from scratch using PyTorch.

**Actor**
```
18 → 128 → 128 → 6
```
The Actor receives local observations and produces a probability distribution over the six actions.

**Central Critic**
```
73 → 128 → 128 → 10
```
The Critic receives the global state and estimates the value for each of the 10 drones.

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
│   ├── networks.py
│   └── rollout.py
│
├── checkpoints/
│   ├── mappo.pt
│   └── mappo_best.pt
│
├── train_mappo.py
├── evaluate_mappo.py
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
| `observations.py` | Local observations and global state |
| `movement.py` | Movement, battery and action masking |
| `collisions.py` | Multi-drone collision detection |
| `rewards.py` | Reward calculation |
| `renderer.py` | Environment visualization |
| `networks.py` | Actor and Central Critic |
| `rollout.py` | GAE and return calculation |
| `agent.py` | MAPPO training logic |
| `train_mappo.py` | Training pipeline |
| `evaluate_mappo.py` | Model evaluation |
| `test_random_city.py` | Random city generation testing |

---

## 🧪 Testing

The project includes `test_random_city.py` to verify the random city generation.

Run:

```bash
python test_random_city.py
```

The test helps verify that the environment can generate different city configurations using different random seeds. This supports the use of randomized environments during training and testing.

---

## 🚀 Installation

```bash
git clone <your-repository-url>

cd Autonomous-Drone-Traffic-Optimization

python3.12 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Training

Train the MAPPO agent:

```bash
python train_mappo.py
```

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

The trained models are saved in:

```
checkpoints/
```

---

## 📊 Evaluation

Evaluate a trained model using:

```bash
python evaluate_mappo.py --model checkpoints/mappo_best.pt --episodes 10 --render
```

The evaluation tracks:

- Delivery rate
- Total deliveries
- Collisions
- No-fly violations
- Battery remaining
- Episode reward
- Steps per episode

---

## 📌 Current Configuration

| Setting | Value |
|---|---|
| Number of drones | 10 |
| Local observation | 18 |
| Global state | 73 |
| Actions per drone | 6 |
| Actor architecture | 18 → 128 → 128 → 6 |
| Critic architecture | 73 → 128 → 128 → 10 |
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
