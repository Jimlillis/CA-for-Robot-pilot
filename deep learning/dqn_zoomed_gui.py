import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import matplotlib.pyplot as plt

grid_size = 200
zoom_size = 20  # Περιοχή zoom γύρω από πράκτορα
cell_size = 20
width = height = zoom_size * cell_size
goal = (grid_size - 1, grid_size - 1)
num_episodes = 200
max_steps = 500
obstacle_count = int(grid_size * grid_size * 0.10)

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 0.0005

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)
GREY  = (180, 180, 180)

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
    def forward(self, x):
        return self.fc(x)

model = DQN()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

def get_state_tensor(state):
    return torch.FloatTensor([state[0] / grid_size, state[1] / grid_size])

def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, 3)
    with torch.no_grad():
        q_values = model(get_state_tensor(state))
        return torch.argmax(q_values).item()

def get_next_state(state, action, grid):
    dx, dy = actions[action]
    nx, ny = state[0] + dx, state[1] + dy
    if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx, ny] != -1:
        return (nx, ny)
    return state

def get_reward(state, grid):
    if state == goal:
        return 50
    elif grid[state] == -1:
        return -10
    else:
        return -0.05

def create_grid():
    grid = np.zeros((grid_size, grid_size), dtype=int)
    grid[goal] = 2
    count = 0
    while count < obstacle_count:
        x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        if (x, y) != (0, 0) and (x, y) != goal and grid[x, y] == 0:
            grid[x, y] = -1
            count += 1
    return grid

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("DQN Grid 200x200 - Zoom View")

def draw_zoom(grid, robot_pos):
    center_x, center_y = robot_pos
    half = zoom_size // 2
    start_x = max(center_x - half, 0)
    start_y = max(center_y - half, 0)

    for i in range(zoom_size):
        for j in range(zoom_size):
            grid_x = start_x + i
            grid_y = start_y + j
            if grid_x >= grid_size or grid_y >= grid_size:
                continue
            color = WHITE
            if grid[grid_x, grid_y] == -1:
                color = BLACK
            elif grid[grid_x, grid_y] == 2:
                color = GREEN
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GREY, rect, 1)

    rx = center_x - start_x
    ry = center_y - start_y
    pygame.draw.rect(screen, BLUE, (ry * cell_size, rx * cell_size, cell_size, cell_size))

episode_rewards = []
for episode in range(num_episodes):
    grid = create_grid()
    state = (0, 0)
    total_reward = 0

    for step in range(max_steps):
        if episode % 5 == 0:
            pygame.time.delay(1)
            screen.fill(WHITE)
            draw_zoom(grid, state)
            pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = choose_action(state)
        next_state = get_next_state(state, action, grid)
        reward = get_reward(next_state, grid)
        total_reward += reward

        with torch.no_grad():
            target = reward + gamma * torch.max(model(get_state_tensor(next_state)))

        output = model(get_state_tensor(state))[action]
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if state == goal:
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

pygame.quit()

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Performance with Zoom View")
plt.grid()
plt.show()
