# Cellular Automaton Robot Navigation

A semi-annual project (Group 19) implementing autonomous robot navigation on a grid using **Cellular Automaton (CA) enhanced Q-learning** and a **Deep Q-Network (DQN)** variant.

The robot starts at `(0, 0)` and must reach the bottom-right corner `(grid_size-1, grid_size-1)` while avoiding randomly placed obstacles. What makes this different from standard Q-learning is the **cellular automaton rule**: after each Q-value update, the knowledge propagates (5% soft update) to all adjacent cells, allowing spatial learning to spread across the grid organically.

---

## Features

- **CA Q-Learning** — Each grid cell owns its own Q-values. After every step, updated values diffuse to von Neumann neighbors.
- **Pheromone layer** — Cells accumulate and decay pheromones as the robot visits them (inspired by ant colony optimization), visualized as a green intensity overlay.
- **Live pygame visualization** — Zoomed viewport centered on the robot with a real-time info panel (episode, epsilon, success rate, steps).
- **Post-training analysis** — 6-panel matplotlib dashboard: rewards per episode, success rate (moving average), steps per episode, normalized Q-value heatmap, successes per run, exploration efficiency.
- **DQN variant** — Neural network (3-layer MLP via PyTorch) on a 200×200 grid with zoomed pygame view.

---

## Requirements

```
Python 3.8+
numpy
pygame
matplotlib
seaborn
```

Install all at once:
```bash
pip install -r requirements.txt
```

For the DQN variant, also install PyTorch:
```bash
pip install torch
```

---

## Running the Project

### Main script (recommended — fully self-contained)

```bash
python "Κυψελιδωτό αυτόματο.py"
```

Trains the CA Q-learning agent on a 30×30 grid with 20% obstacle density for 300 episodes, then displays the analysis charts.

---

### Modular variants (inside `CA-for-Robot-pilot-/`)

Each variant separates training from simulation. Run from inside the variant's folder:

#### `final/` — Train then simulate (two separate steps)
```bash
cd "CA-for-Robot-pilot-/final"
python run_all.py
```
`run_all.py` runs `main.py` (trains Q-table, saves `q_table.npy` and `grid.npy`), then launches `pygame_simulation.py` which loads the saved files and replays the learned path.

**Tip:** In the simulation window you can left-click to add obstacles and right-click to remove them. Press `R` to reset.

#### `final_livelearn/` — Train and visualize simultaneously
```bash
cd "CA-for-Robot-pilot-/final_livelearn"
python run_all.py
```
Trains and renders at the same time. Press `R` in the pygame window to reset learning.

#### `deep learning/` — DQN on a 200×200 grid
```bash
cd "CA-for-Robot-pilot-/deep learning"
python dqn_zoomed_gui.py
```
Uses a 3-layer MLP (input: normalized `(x, y)` coordinates → 256 → 256 → 4 Q-values). Requires PyTorch.

---

## Configurable Hyperparameters

All parameters are set in the `Config` dataclass at the top of each main script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | 30 | Grid dimensions (N×N) |
| `obstacle_density` | 0.20 | Fraction of cells that are obstacles |
| `max_episodes` | 300 | Training episodes |
| `alpha` | 0.15 | Q-learning rate |
| `gamma` | 0.9 | Discount factor |
| `epsilon` | 0.5 | Initial exploration rate |
| `epsilon_decay` | 0.995 | Epsilon decay per episode |
| `epsilon_min` | 0.01 | Minimum exploration rate |
| `goal_reward` | 100.0 | Reward for reaching the goal |
| `obstacle_penalty` | -10.0 | Penalty for hitting an obstacle |
| `visited_penalty` | -0.5 | Penalty for revisiting a cell |

---

## Project Structure

```
.
├── Κυψελιδωτό αυτόματο.py     # Main standalone script (start here)
├── requirements.txt
└── CA-for-Robot-pilot-/
    ├── environment.py              # Basic grid env with BFS utility
    ├── q_learning.py               # Early Q-learning prototype
    ├── SemiAnnualF.py              # Full CA implementation (same as main script)
    ├── final/                      # Train → save → simulate pipeline
    │   ├── environment.py
    │   ├── main.py                 # Trains Q-table, saves .npy files
    │   ├── pygame_simulation.py    # Loads .npy files, plays back path
    │   └── run_all.py
    ├── final_livelearn/            # Live training + visualization
    │   ├── environment_live.py
    │   ├── main_live.py
    │   ├── pygame_simulation_live.py
    │   └── run_all.py
    └── deep learning/
        └── dqn_zoomed_gui.py       # DQN agent with PyTorch
```

---

## How the Cellular Automaton Works

Standard Q-learning updates only the cell the robot just left. This project adds a **spatial propagation step**: after each update, the cell's new Q-values are blended into all 4 neighbors at a 5% rate:

```python
neighbor.q_values = 0.95 * neighbor.q_values + 0.05 * current_cell.q_values
```

This causes "good" path knowledge to spread outward over time, helping the agent generalize across nearby states without visiting them directly.

---

## Authors

Group 19 — Semester Project  
Students: Dimitris Lillis, Anthonis Themelis
