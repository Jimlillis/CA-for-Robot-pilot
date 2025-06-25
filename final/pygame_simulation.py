import pygame
import numpy as np
import time

# Load saved data
Q = np.load('q_table.npy')
grid = np.load('grid.npy')

# Parameters
grid_size = grid.shape[0]
cell_size = 50
width = height = grid_size * cell_size
fps = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)
GREY  = (200, 200, 200)

# Actions
actions = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

def get_next_state(state):
    x, y = state
    action = np.argmax(Q[x, y])
    dx, dy = actions[action]
    nx, ny = x + dx, y + dy
    if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx, ny] != -1:
        return (nx, ny)
    return (x, y)

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

    # Draw robot
    x, y = robot_pos
    robot_rect = pygame.Rect(y*cell_size+5, x*cell_size+5, cell_size-10, cell_size-10)
    pygame.draw.rect(screen, BLUE, robot_rect)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Robot Path Simulation")
clock = pygame.time.Clock()

robot_pos = (0, 0)
goal = tuple(map(int, np.argwhere(grid == 10)[0]))
path = []
running = True
paused = False

while running:
    clock.tick(fps)
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            j, i = mx // cell_size, my // cell_size
            if (i, j) != robot_pos and (i, j) != goal:
                if event.button == 1 and grid[i, j] == 0:
                    grid[i, j] = -1
                elif event.button == 3 and grid[i, j] == -1:
                    grid[i, j] = 0

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                robot_pos = (0, 0)
                path = []
                paused = False

    if not paused:
        if robot_pos == goal:
            paused = True
        else:
            path.append(robot_pos)
            robot_pos = get_next_state(robot_pos)

    draw_grid(robot_pos, path)
    pygame.display.flip()

pygame.quit()
