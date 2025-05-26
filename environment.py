import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import random
from collections import deque

# Μέγεθος πλέγματος
GRID_SIZE = 10

# Δημιουργία πλέγματος
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Εμπόδια χωρίς να περιλαμβανουν αρχη / στόχο
excluded = {(0, 0), (9, 9)}
candidates = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if (i, j) not in excluded]
obstacles = random.sample(candidates, 20)  # Επιλογή 20 τυχαίων εμποδίων
for (x, y) in obstacles:
    grid[x][y] = 1

# Θέσεις
start_pos = (0, 0)
goal_pos = (9, 9)
robot_pos = start_pos

# Ενέργειες
actions = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Κίνηση ρομπότ
def move_robot(pos, action, grid):
    dx, dy = actions[action]
    new_x, new_y = pos[0] + dx, pos[1] + dy
    
    if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
        print("Εκτός ορίων!")
        return pos
    if grid[new_x][new_y] == 1:
        print("Εμπόδιο!")
        return pos
    return (new_x, new_y)

# Οπτικοποίηση
def visualize(grid, robot_pos, goal_pos):
    display_grid = grid.copy()
    x, y = robot_pos
    gx, gy = goal_pos
    
    # Κώδικες:
    # 0: ελεύθερο (λευκό)
    # 1: εμπόδιο (μαύρο)
    # 2: ρομπότ (μπλε)
    # 3: στόχος (πράσινο)
    display_grid[x][y] = 2
    display_grid[gx][gy] = 3
    
    cmap = colors.ListedColormap(['white', 'black', 'blue', 'green'])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(display_grid, cmap=cmap, norm=norm)
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    plt.xticks(np.arange(-.5, GRID_SIZE, 1), [])
    plt.yticks(np.arange(-.5, GRID_SIZE, 1), [])
    plt.pause(0.5)
    plt.clf()


def bfs_path(grid, start, goal):
    queue = deque()
    queue.append((start, [start]))
    visited = set()
    visited.add(start)
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path
        for dx, dy in actions.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if grid[nx][ny] != 1 and (nx, ny) not in visited:
                    queue.append(((nx, ny), path + [(nx, ny)]))
                    visited.add((nx, ny))
    return None  # Δεν βρέθηκε διαδρομή

# Ενεργοποίηση γραφικών
plt.figure(figsize=(6,6))


# Βρες διαδρομή
path = bfs_path(grid, start_pos, goal_pos)

if path:
    for pos in path[1:]:
        robot_pos = pos
        visualize(grid, robot_pos, goal_pos)
    print(f"Τελική θέση ρομπότ: {robot_pos}")
    visualize(grid, robot_pos, goal_pos)
else:
    print("Δεν βρέθηκε διαδρομή προς τον στόχο.")
    
# Ενεργοποίηση real-time γραφικών
plt.figure(figsize=(6,6))

# Παράδειγμα διαδοχικών κινήσεων
commands = ['right', 'right', 'down', 'down', 'down', 'right', 'down', 'right', 'down', 'down']

for cmd in commands:
    robot_pos = move_robot(robot_pos, cmd, grid)
    visualize(grid, robot_pos, goal_pos)
    # Τερματισμός αν φτάσουμε στον στόχο
    if robot_pos == goal_pos:
        print("Φτάσαμε στον στόχο!")
        break
plt.close()

# Εμφάνιση τελικής κατάστασης
visualize(grid, robot_pos, goal_pos)

