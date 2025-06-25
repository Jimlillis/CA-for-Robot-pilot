import numpy as np
import random

class GridEnvironment:
    def __init__(self, size=10, num_obstacles=15):
        self.size = size
        self.grid = np.zeros((size, size))
        self.num_obstacles = num_obstacles
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = []

        self._generate_random_environment()

    def _generate_random_environment(self):
        # Ορισμός στόχου
        self.grid[self.goal] = 10

        # Προσθήκη εμποδίων (όχι στην αρχή και στο στόχο)
        count = 0
        while count < self.num_obstacles:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) != self.start and (x, y) != self.goal and self.grid[x, y] == 0:
                self.grid[x, y] = -1
                self.obstacles.append((x, y))
                count += 1

    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[x, y] != -1

    def get_reward(self, pos):
        if pos == self.goal:
            return 10
        elif self.grid[pos] == -1:
            return -10
        else:
            return -1

    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.obstacles = []
        self._generate_random_environment()
