Absolutely. I’ll keep your existing README **exactly in the same style and structure**, and add `test_random_city.py` everywhere it belongs.

Here is the **complete updated `README.md`**, ready to copy-paste:

````markdown
# 🚁 Autonomous Drone Traffic Optimization

A **Multi-Agent Reinforcement Learning (MARL)** system that trains autonomous drones to perform delivery tasks while avoiding obstacles, collisions, no-fly zones, and battery failures.

The project uses **MAPPO (Multi-Agent Proximal Policy Optimization)** with **Centralized Training and Decentralized Execution (CTDE)**.

---

# 🎯 Project Overview

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

# 🧠 Core Approach

The system follows **Centralized Training and Decentralized Execution (CTDE)**.

```text
                    TRAINING

                       │
                       ▼
              Global Environment
                       │
                       ▼
               ┌───────────────┐
               │ Central Critic│
               └───────┬───────┘
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
              ┌────────┼────────┐
              ▼        ▼        ▼
           Drone 1  Drone 2  Drone 10
              │        │        │
              ▼        ▼        ▼
          Local Obs Local Obs Local Obs
              │        │        │
              ▼        ▼        ▼
           Action   Action   Action
````

### During Training

The **Central Critic** has access to global information about all drones.

### During Execution

Each drone uses only its **local observation** to make decisions.

This allows the trained system to operate without requiring every drone to continuously access the complete global environment state.

---

# 🏙️ Simulation Environment

A custom **Gymnasium** environment represents a small city containing:

```text
🏢 Buildings

🚫 No-Fly Zones

📦 Warehouses

📍 Delivery Points

⚡ Charging Stations

🚁 10 Drones
```

The environment supports seeded random generation, allowing different scenarios to be created while maintaining reproducibility.

---

# 🚁 Drone Tasks

Each drone receives a delivery task:

```text
Warehouse → Delivery Point
```

During an episode, the drone must navigate from its source to its assigned destination while handling environmental constraints and other drones.

---

# 🎮 Action Space

Each drone can choose one of six actions:

| Action | Description |
| ------ | ----------- |
| `0`    | Move Up     |
| `1`    | Move Down   |
| `2`    | Move Left   |
| `3`    | Move Right  |
| `4`    | Wait        |
| `5`    | Recharge    |

All 10 drones select actions simultaneously.

---

# 👀 Local Observation

Each drone receives an **18-dimensional local observation** containing information such as:

* Relative destination
* Battery level
* Nearby obstacles
* Nearby drones
* Nearest-drone information
* Local traffic density
* Charging station availability
* Delivery status

```text
10 drones × 18 local features
```

The same Actor network is shared across all drones.

---

# 🌐 Global State

During training, the Central Critic receives a global representation of the environment.

For 10 drones:

```text
10 × 7 drone features + 3 global features
= 73-dimensional global state
```

The Critic produces:

```text
10 value estimates
```

one for each drone.

---

# 🧮 Reward System

The reward function encourages successful, safe, and efficient deliveries.

| Event                       | Reward |
| --------------------------- | -----: |
| Successful delivery         |   +100 |
| Collision                   |    -80 |
| Illegal action              |     -2 |
| Battery failure             |    -40 |
| Progress toward destination |     +2 |
| Waiting                     |  -0.10 |
| Each step                   |  -0.05 |

The objective is therefore not simply to reach the destination, but to do so **safely and efficiently**.

---

# 🛡️ Safety Mechanisms

The environment includes several safety mechanisms.

### Action Masking

Invalid actions are prevented from being selected.

Examples:

* Moving outside the city
* Entering buildings
* Entering no-fly zones
* Moving without sufficient battery
* Recharging away from a charging station

### Collision Detection

The system detects:

* Two drones occupying the same cell
* Two drones swapping positions in the same timestep

---

# 🤖 MAPPO Architecture

The project implements MAPPO from scratch using PyTorch.

### Actor

```text
18 → 128 → 128 → 6
```

The Actor receives local observations and produces a probability distribution over the six actions.

### Central Critic

```text
73 → 128 → 128 → 10
```

The Critic receives the global state and estimates the value for each of the 10 drones.

---

# 📈 PPO + GAE

The MAPPO implementation uses:

* PPO clipped objective
* Advantage normalization
* Entropy regularization
* Gradient clipping
* Mini-batch updates
* Generalized Advantage Estimation (GAE)

GAE uses rewards and Critic value estimates to calculate:

```text
Advantages → Actor update

Returns    → Critic update
```

This helps provide more stable policy learning.

---

# 📂 Project Structure

```text
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

# 🔧 Main Components

| File                  | Purpose                              |
| --------------------- | ------------------------------------ |
| `drone_env.py`        | Main Gymnasium environment           |
| `city.py`             | City and obstacle generation         |
| `drone.py`            | Drone state and tracking             |
| `obstacles.py`        | Building and no-fly zone structures  |
| `task_generator.py`   | Delivery task generation             |
| `observations.py`     | Local observations and global state  |
| `movement.py`         | Movement, battery and action masking |
| `collisions.py`       | Multi-drone collision detection      |
| `rewards.py`          | Reward calculation                   |
| `renderer.py`         | Environment visualization            |
| `networks.py`         | Actor and Central Critic             |
| `rollout.py`          | GAE and return calculation           |
| `agent.py`            | MAPPO training logic                 |
| `train_mappo.py`      | Training pipeline                    |
| `evaluate_mappo.py`   | Model evaluation                     |
| `test_random_city.py` | Random city generation testing       |

---

# 🧪 Testing

The project includes `test_random_city.py` to verify the random city generation.

Run:

```bash
python test_random_city.py
```

The test helps verify that the environment can generate different city configurations using different random seeds.

This supports the use of randomized environments during training and testing.

---

# 🚀 Installation

```bash
git clone <your-repository-url>

cd Autonomous-Drone-Traffic-Optimization

python3.12 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

# ▶️ Training

Train the MAPPO agent:

```bash
python train_mappo.py
```

The training process:

```text
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

```text
checkpoints/
```

---

# 📊 Evaluation

Evaluate a trained model using:

```bash
python evaluate_mappo.py --model checkpoints/mappo_best.pt --episodes 10 --render
```

The evaluation tracks:

* Delivery rate
* Total deliveries
* Collisions
* No-fly violations
* Battery remaining
* Episode reward
* Steps per episode

---

# 📌 Current Configuration

```text
Number of drones       : 10

Local observation      : 18

Global state           : 73

Actions per drone      : 6

Actor architecture     : 18 → 128 → 128 → 6

Critic architecture    : 73 → 128 → 128 → 10

Algorithm              : MAPPO

Training paradigm      : CTDE

Environment             : Gymnasium

Deep Learning           : PyTorch
```

---

# 🔬 Future Work

Potential extensions include:

* 🌐 Graph Neural Networks for drone interaction
* 📡 Communication-aware multi-agent policies
* 🗺️ Larger and more realistic city maps
* 🌦️ Weather and wind simulation
* 🔋 More realistic battery models
* 🚁 Continuous 2D/3D drone movement
* 📦 Dynamic task assignment
* 📊 Baseline comparison with other MARL algorithms
* 📈 Large-scale fleet experiments

---

# 🛠️ Technologies

Python

PyTorch

Gymnasium

NumPy

Pygame

MAPPO

PPO

GAE

Multi-Agent Reinforcement Learning


---

# 📜 Project Status

🚧 **Research / Development Project**

The current implementation provides the complete foundation for a multi-drone MARL system, including environment simulation, local observations, centralized training, decentralized execution, collision handling, battery management, MAPPO training, randomized city testing, and evaluation.

The next stage focuses on improving scalability, generalization, collision avoidance, and experimental evaluation.

---

# ⭐ Key Takeaway

> **Train globally. Execute locally. Coordinate intelligently.**

This project explores how multiple autonomous drones can learn safe and efficient delivery behavior through **Multi-Agent Reinforcement Learning** rather than relying entirely on manually designed navigation rules.



I also added `obstacles.py` to the structure/component table because it is actually part of your current project structure, so the README now reflects the files you have rather than leaving one out. Your original README already establishes the 10-drone MAPPO/CTDE architecture and 73-dimensional global state. :contentReference[oaicite:0]{index=0}

