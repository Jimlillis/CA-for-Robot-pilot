import numpy as np
import random
import matplotlib.pyplot as plt
from environment import grid, start_pos, goal_pos, move_robot, visualize, GRID_SIZE

# --- Q-Learning Παράμετροι ---
alpha = 0.1        # Learning rate
gamma = 0.9        # Discount factor
epsilon = 0.2      # Exploration rate
episodes = 300     # Πλήθος επεισοδίων

# Χάρτης ενεργειών
action_names = ['up', 'down', 'left', 'right']
action_map = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

# Πίνακας Q
q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))

# Επιλογή δράσης (με ερευνητική στρατηγική)
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        x, y = state
        return np.argmax(q_table[x][y])

# Έλεγχος εγκυρότητας θέσης
def is_valid(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x][y] != 1

# Εκπαίδευση
for ep in range(episodes):
    state = start_pos
    robot_pos = state
    steps = 0

    while robot_pos != goal_pos and steps < 100:
        x, y = robot_pos
        action_idx = choose_action(robot_pos)
        dx, dy = action_map[action_idx]
        new_pos = (x + dx, y + dy)

        if not is_valid(new_pos):
            reward = -10
            new_pos = robot_pos
        elif new_pos == goal_pos:
            reward = 100
        else:
            reward = -1

        nx, ny = new_pos
        old_q = q_table[x][y][action_idx]
        next_max = np.max(q_table[nx][ny])
        q_table[x][y][action_idx] = old_q + alpha * (reward + gamma * next_max - old_q)

        robot_pos = new_pos
        steps += 1

    if ep % 50 == 0:
        print(f"Επεισόδιο {ep} ολοκληρώθηκε σε {steps} βήματα.")

print("Εκπαίδευση Ολοκληρώθηκε")

# --- Αναπαραγωγή διαδρομής που έμαθε το Q-table ---

robot_pos = start_pos
path = [robot_pos]

plt.figure(figsize=(6, 6))
max_steps = 100
for _ in range(max_steps):
    x, y = robot_pos
    action_idx = np.argmax(q_table[x][y])
    dx, dy = action_map[action_idx]
    new_pos = (x + dx, y + dy)

    if not is_valid(new_pos):
        print(" Πήγε σε άκυρη θέση, τερματισμός.")
        break

    robot_pos = new_pos
    path.append(robot_pos)
    visualize(grid, robot_pos, goal_pos)

    if robot_pos == goal_pos:
        print(" Έφτασε στον στόχο!")
        break

plt.close()
# Εμφάνιση της διαδρομής που έμαθε το Q-table
plt.figure(figsize=(6, 6))