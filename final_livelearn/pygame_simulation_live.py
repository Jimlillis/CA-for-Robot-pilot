import pygame
import numpy as np
import time
import random

# Parameters
grid_size = 10
cell_size = 50
width = height = grid_size * cell_size
fps = 15
num_obstacles = random.randint(10, 25)

# Learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)
GREY  = (200, 200, 200)
RED   = (255, 0, 0)

# Actions
actions = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1)    # right
}

# Initialize grid
grid = np.zeros((grid_size, grid_size))
start = (0, 0)
goal = (grid_size - 1, grid_size - 1)
grid[goal] = 10
obstacles = 0
while obstacles < num_obstacles:
    x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
    if (x, y) not in [start, goal] and grid[x, y] == 0:
        grid[x, y] = -1
        obstacles += 1

# Initialize Q-table
Q = np.zeros((grid_size, grid_size, len(actions)))

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(list(actions.keys()))
    else:
        return np.argmax(Q[state[0], state[1]])

def get_next_state(state, action):
    dx, dy = actions[action]
    nx, ny = state[0] + dx, state[1] + dy
    if 0 <= nx < grid_size and 0 <= ny < grid_size:
        return (nx, ny)
    return state

def get_reward(state):
    if grid[state] == -1:
        return -10
    elif state == goal:
        return 10
    else:
        return -1

def draw_grid(robot_pos, path):
    for i in range(grid_size):
        for j in range(grid_size):
            rect = pygame.Rect(j*cell_size, i*cell_size, cell_size, cell_size)
            if grid[i, j] == -1:
                pygame.draw.rect(screen, BLACK, rect)
            elif grid[i, j] == 10:
                pygame.draw.rect(screen, GREEN, rect)
            elif (i, j) in path:
                pygame.draw.rect(screen, GREY, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

    x, y = robot_pos
    robot_rect = pygame.Rect(y*cell_size+5, x*cell_size+5, cell_size-10, cell_size-10)
    pygame.draw.rect(screen, BLUE, robot_rect)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Robot Path - Online Q-Learning")
clock = pygame.time.Clock()

robot_pos = start
path = []
running = True
paused = False

while running:
    clock.tick(fps)
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                robot_pos = start
                path = []
                Q = np.zeros((grid_size, grid_size, len(actions)))  # reset learning

    if not paused:
        action = choose_action(robot_pos)
        next_state = get_next_state(robot_pos, action)
        reward = get_reward(next_state)
        best_next = np.max(Q[next_state[0], next_state[1]])
        Q[robot_pos[0], robot_pos[1], action] += alpha * (reward + gamma * best_next - Q[robot_pos[0], robot_pos[1], action])

        if grid[next_state] != -1:
            path.append(robot_pos)
            robot_pos = next_state

        if robot_pos == goal:
            paused = True

    draw_grid(robot_pos, path)
    pygame.display.flip()

pygame.quit()
