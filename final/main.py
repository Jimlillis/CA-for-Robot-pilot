import numpy as np
import matplotlib.pyplot as plt
import random
from environment import GridEnvironment

# Parameters
grid_size = 10
num_episodes = 500
gamma = 0.9
alpha = 0.1
epsilon = 0.2

# Create new random environment each run
env = GridEnvironment(size=grid_size, num_obstacles=random.randint(10, 25))

# Actions: up, down, left, right
actions = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

# Initialize Q-table
Q = np.zeros((grid_size, grid_size, len(actions)))

# CA State Matrix
CA = np.zeros((grid_size, grid_size))

trajectory = []

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(actions.keys()))
    else:
        return np.argmax(Q[state[0], state[1]])

def get_next_state(state, action):
    dx, dy = actions[action]
    next_state = (state[0] + dx, state[1] + dy)
    if env.is_valid(next_state):
        return next_state
    return state

def update_ca():
    global CA
    new_CA = np.copy(CA)
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            neighborhood = CA[i - 1:i + 2, j - 1:j + 2]
            new_CA[i, j] = 0.1 * np.sum(neighborhood)
    CA = new_CA

# Training loop
for episode in range(num_episodes):
    state = env.start
    if episode == num_episodes - 1:
        trajectory = []
    while state != env.goal:
        action = choose_action(state)
        next_state = get_next_state(state, action)
        reward = env.get_reward(next_state)
        best_next_action = np.max(Q[next_state[0], next_state[1]])
        Q[state[0], state[1], action] += alpha * (reward + gamma * best_next_action - Q[state[0], state[1], action])
        CA[state[0], state[1]] += 1
        if episode == num_episodes - 1:
            trajectory.append(state)
        state = next_state
    update_ca()

# Save Q-table and grid for simulation
np.save("q_table.npy", Q)
np.save("grid.npy", env.grid)

# Plot trajectory
fig, ax = plt.subplots()
grid_display = np.copy(env.grid)
for x, y in trajectory:
    if (x, y) != env.start and (x, y) != env.goal and env.grid[x, y] != -1:
        grid_display[x, y] = 5

cmap = plt.cm.get_cmap("coolwarm", 11)
ax.imshow(grid_display, cmap=cmap, interpolation='nearest')
ax.set_title("Final Trajectory to Goal")
plt.colorbar(ax.imshow(grid_display, cmap=cmap, interpolation='nearest'), ax=ax)
plt.show()

# Plot CA activity
plt.imshow(CA, cmap='hot', interpolation='nearest')
plt.title("Cellular Automata Activity Map")
plt.colorbar(label="Activity Level")
plt.show()
