import numpy as np
import pygame
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import json
import time
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque, defaultdict

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CellState(Enum):
    """Καταστάσεις κελιών του Κυψελιδωτού Αυτομάτου"""
    EMPTY = 0
    OBSTACLE = -1
    GOAL = 2
    ROBOT = 3
    VISITED = 1

class ActionType(Enum):
    """Δυνατές ενέργειες του ρομπότ"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

@dataclass
class Config:
    """Παράμετροι συστήματος"""
    # Grid parameters
    grid_size: int = 50
    obstacle_density: float = 0.15
    
    # Visualization parameters
    cell_size: int = 12
    zoom_size: int = 25
    fps: int = 30
    
    # CA Learning parameters
    alpha: float = 0.1  # Learning rate
    gamma: float = 0.9  # Discount factor
    epsilon: float = 0.3  # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    
    # Training parameters
    max_episodes: int = 500
    max_steps_per_episode: int = 1000
    
    # Rewards
    goal_reward: float = 100.0
    obstacle_penalty: float = -10.0
    step_penalty: float = -0.1
    visited_penalty: float = -0.5

class Cell:
    """Κελί του Κυψελιδωτού Αυτομάτου"""
    
    def __init__(self, x: int, y: int, state: CellState = CellState.EMPTY):
        self.x = x
        self.y = y
        self.state = state
        self.q_values = np.zeros(4)  # Q-values για κάθε ενέργεια
        self.visit_count = 0
        self.pheromone = 0.0  # Φερομόνες για swarm intelligence
        self.neighbors: List['Cell'] = []
        
    def add_neighbor(self, neighbor: 'Cell'):
        """Προσθήκη γείτονα"""
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
    
    def get_best_action(self) -> int:
        """Επιστρέφει την καλύτερη ενέργεια βάσει Q-values"""
        return np.argmax(self.q_values)
    
    def update_q_value(self, action: int, reward: float, next_max_q: float, alpha: float, gamma: float):
        """Ενημέρωση Q-value για συγκεκριμένη ενέργεια"""
        old_q = self.q_values[action]
        self.q_values[action] = old_q + alpha * (reward + gamma * next_max_q - old_q)
    
    def update_pheromone(self, decay_rate: float = 0.95):
        """Ενημέρωση φερομονών"""
        self.pheromone *= decay_rate

class CellularAutomatonGrid:
    """Κυψελιδωτό Αυτόματο για πλοήγηση ρομπότ"""
    
    def __init__(self, config: Config):
        self.config = config
        self.size = config.grid_size
        self.grid: List[List[Cell]] = []
        self.robot_pos = (0, 0)
        self.goal_pos = (config.grid_size - 1, config.grid_size - 1)
        self.visited_cells = set()
        self.path_history = []
        # Προσθήκη λιστών στατιστικών
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []
        # Απόδοση ανά run
        self.run_performance = {
            'successful_episodes': [],
            'average_rewards': [],
            'exploration_efficiency': []
        }
        
    def _initialize_grid(self):
        """Αρχικοποίηση του πλέγματος"""
        # Δημιουργία κελιών
        for i in range(self.size):
            row = []
            for j in range(self.size):
                cell = Cell(i, j)
                row.append(cell)
            self.grid.append(row)
        
        # Τοποθέτηση εμποδίων
        num_obstacles = int(self.size * self.size * self.config.obstacle_density)
        obstacles_placed = 0
        
        while obstacles_placed < num_obstacles:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            
            if (x, y) not in [(0, 0), self.goal_pos] and self.grid[x][y].state == CellState.EMPTY:
                self.grid[x][y].state = CellState.OBSTACLE
                obstacles_placed += 1
        
        # Θέση στόχου
        self.grid[self.goal_pos[0]][self.goal_pos[1]].state = CellState.GOAL
        
    def _setup_neighbors(self):
        """Σύνδεση γειτόνων για κάθε κελί"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i][j]
                
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        neighbor = self.grid[nx][ny]
                        cell.add_neighbor(neighbor)
    
    def get_valid_actions(self, pos: Tuple[int, int]) -> List[int]:
        """Επιστρέφει έγκυρες ενέργειες από τη δεδομένη θέση"""
        x, y = pos
        valid_actions = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for action, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.size and 0 <= ny < self.size and 
                self.grid[nx][ny].state != CellState.OBSTACLE):
                valid_actions.append(action)
        
        return valid_actions

    def move_robot(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Κίνηση ρομπότ και επιστροφή νέας θέσης, ανταμοιβής και done flag"""
        x, y = self.robot_pos
        
        # Έλεγχος ορίων για τρέχουσα θέση
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            self.robot_pos = (0, 0)
            x, y = 0, 0
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Έλεγχος έγκυρης ενέργειας
        if action < 0 or action >= len(directions):
            action = 0
            
        dx, dy = directions[action]
        new_x, new_y = x + dx, y + dy
        
        # Έλεγχος ορίων και εμποδίων
        if (0 <= new_x < self.size and 0 <= new_y < self.size and 
            self.grid[new_x][new_y].state != CellState.OBSTACLE):
            
            # Ενημέρωση θέσης
            old_pos = self.robot_pos
            self.robot_pos = (new_x, new_y)
            self.path_history.append(self.robot_pos)
            
            # Υπολογισμός ανταμοιβής
            reward = self._calculate_reward(self.robot_pos, old_pos)
            
            # Έλεγχος επίτευξης στόχου
            done = self.robot_pos == self.goal_pos
            
            # Ενημέρωση visited cells
            if self.robot_pos not in self.visited_cells:
                self.visited_cells.add(self.robot_pos)
                self.grid[new_x][new_y].visit_count += 1
            
            # Ενημέρωση φερομονών
            self.grid[new_x][new_y].pheromone += 1.0
            
            return self.robot_pos, reward, done
        else:
            # Άκυρη κίνηση - μένουμε στη θέση μας
            return self.robot_pos, self.config.obstacle_penalty, False

    def _calculate_reward(self, current_pos: Tuple[int, int], previous_pos: Tuple[int, int]) -> float:
        """Υπολογισμός ανταμοιβής"""
        x, y = current_pos
        cell = self.grid[x][y]
        
        # Βασική ανταμοιβή
        if current_pos == self.goal_pos:
            return self.config.goal_reward
        elif cell.state == CellState.OBSTACLE:
            return self.config.obstacle_penalty
        elif current_pos in self.visited_cells:
            return self.config.visited_penalty
        
        # Ανταμοιβή προόδου (πλησίασμα στο στόχο)
        old_distance = np.sqrt((previous_pos[0] - self.goal_pos[0])**2 + 
                              (previous_pos[1] - self.goal_pos[1])**2)
        new_distance = np.sqrt((current_pos[0] - self.goal_pos[0])**2 + 
                              (current_pos[1] - self.goal_pos[1])**2)
        
        progress_reward = (old_distance - new_distance) * 0.5
        
        return self.config.step_penalty + progress_reward

    def choose_action(self, pos: Tuple[int, int], epsilon: float) -> int:
        """Επιλογή ενέργειας με ε-greedy policy"""
        x, y = pos
        
        # Έλεγχος ορίων
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            # Αν είμαστε εκτός ορίων, επιστρέφουμε στην αρχή
            self.robot_pos = (0, 0)
            x, y = 0, 0
        
        cell = self.grid[x][y]
        valid_actions = self.get_valid_actions((x, y))
        
        if not valid_actions:
            return 0  # Default action
        
        if np.random.random() < epsilon:
            # Exploration: τυχαία ενέργεια
            return np.random.choice(valid_actions)
        else:
            # Exploitation: καλύτερη ενέργεια
            valid_q_values = [cell.q_values[a] if a in valid_actions else -np.inf 
                            for a in range(4)]
            best_action = np.argmax(valid_q_values)
            return best_action if best_action in valid_actions else valid_actions[0]

    def update_cellular_automaton(self, pos: Tuple[int, int], action: int, reward: float, 
                                next_pos: Tuple[int, int], alpha: float, gamma: float):
        """Ενημέρωση κανόνων του Κυψελιδωτού Αυτομάτου"""
        x, y = pos
        nx, ny = next_pos
        
        current_cell = self.grid[x][y]
        next_cell = self.grid[nx][ny]
        
        # Q-learning update
        next_max_q = np.max(next_cell.q_values)
        current_cell.update_q_value(action, reward, next_max_q, alpha, gamma)
        
        # Ενημέρωση γειτόνων (τοπική διάδοση πληροφορίας)
        for neighbor in current_cell.neighbors:
            if neighbor.state != CellState.OBSTACLE:
                # Soft update των Q-values των γειτόνων
                neighbor.q_values = 0.95 * neighbor.q_values + 0.05 * current_cell.q_values
        
        # Ενημέρωση φερομονών όλων των κελιών
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i][j].update_pheromone()
    
    def reset(self):
        """Επαναφορά περιβάλλοντος για νέο επεισόδιο"""
        self.robot_pos = (0, 0)
        self.visited_cells.clear()
        self.path_history = [(0, 0)]
    
    def get_statistics(self) -> Dict:
        """Επιστροφή στατιστικών"""
        if not self.episode_rewards:
            return {}
        
        return {
            'average_reward': np.mean(self.episode_rewards[-100:]),
            'success_rate': np.mean(self.success_rate[-100:]) if self.success_rate else 0,
            'average_steps': np.mean(self.episode_steps[-100:]) if self.episode_steps else 0,
            'total_episodes': len(self.episode_rewards)
        }

class CAVisualizer:
    """Οπτικοποίηση του Κυψελιδωτού Αυτομάτου"""
    
    def __init__(self, ca_grid: CellularAutomatonGrid):
        self.ca_grid = ca_grid
        self.config = ca_grid.config
        
        # Colors
        self.colors = {
            CellState.EMPTY: (255, 255, 255),      # Λευκό
            CellState.OBSTACLE: (0, 0, 0),         # Μαύρο
            CellState.GOAL: (0, 255, 0),           # Πράσινο
            CellState.ROBOT: (0, 0, 255),          # Μπλε
            CellState.VISITED: (200, 200, 200)     # Γκρι
        }
        
        # Pygame initialization
        pygame.init()
        self.zoom_width = self.config.zoom_size * self.config.cell_size
        self.zoom_height = self.config.zoom_size * self.config.cell_size
        self.info_width = 300
        
        self.screen = pygame.display.set_mode((self.zoom_width + self.info_width, self.zoom_height))
        pygame.display.set_caption("Cellular Automaton Robot Navigation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
    def draw_zoom_view(self):
        """Σχεδίαση zoom view γύρω από το ρομπότ"""
        robot_x, robot_y = self.ca_grid.robot_pos
        half_zoom = self.config.zoom_size // 2
        
        start_x = max(0, robot_x - half_zoom)
        start_y = max(0, robot_y - half_zoom)
        end_x = min(self.ca_grid.size, start_x + self.config.zoom_size)
        end_y = min(self.ca_grid.size, start_y + self.config.zoom_size)
        
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                cell = self.ca_grid.grid[i][j]
                
                # Υπολογισμός χρώματος βάσει κατάστασης και Q-values
                base_color = list(self.colors[cell.state])
                
                # Highlight για visited cells
                if (i, j) in self.ca_grid.visited_cells and cell.state == CellState.EMPTY:
                    base_color = [200, 200, 200]
                
                # Intensity βάσει Q-values (heatmap effect)
                max_q = np.max(cell.q_values)
                if max_q > 0 and cell.state == CellState.EMPTY:
                    intensity = min(255, int(max_q * 10))
                    base_color = [min(255, c + intensity//3) for c in base_color]
                
                # Pheromone visualization
                if cell.pheromone > 0 and cell.state == CellState.EMPTY:
                    pheromone_intensity = min(100, int(cell.pheromone * 20))
                    base_color[1] = min(255, base_color[1] + pheromone_intensity)  # Green channel
                
                # Σχεδίαση κελιού
                screen_x = (j - start_y) * self.config.cell_size
                screen_y = (i - start_x) * self.config.cell_size
                
                rect = pygame.Rect(screen_x, screen_y, self.config.cell_size, self.config.cell_size)
                pygame.draw.rect(self.screen, base_color, rect)
                pygame.draw.rect(self.screen, (128, 128, 128), rect, 1)
        
        # Σχεδίαση ρομπότ
        robot_screen_x = (robot_y - start_y) * self.config.cell_size
        robot_screen_y = (robot_x - start_x) * self.config.cell_size
        robot_rect = pygame.Rect(robot_screen_x + 2, robot_screen_y + 2, 
                               self.config.cell_size - 4, self.config.cell_size - 4)
        pygame.draw.ellipse(self.screen, self.colors[CellState.ROBOT], robot_rect)
    
    def draw_info_panel(self, episode: int, epsilon: float, stats: Dict):
        """Σχεδίαση πάνελ πληροφοριών"""
        info_x = self.zoom_width
        y_offset = 10
        
        # Background
        info_rect = pygame.Rect(info_x, 0, self.info_width, self.zoom_height)
        pygame.draw.rect(self.screen, (240, 240, 240), info_rect)
        
        # Πληροφορίες
        info_texts = [
            f"Episode: {episode}",
            f"Robot Position: {self.ca_grid.robot_pos}",
            f"Goal Position: {self.ca_grid.goal_pos}",
            f"Epsilon: {epsilon:.3f}",
            f"Steps: {len(self.ca_grid.path_history)}",
            f"Visited Cells: {len(self.ca_grid.visited_cells)}",
            "",
            "Statistics:",
            f"Avg Reward: {stats.get('average_reward', 0):.2f}",
            f"Success Rate: {stats.get('success_rate', 0):.2f}",
            f"Avg Steps: {stats.get('average_steps', 0):.1f}",
        ]
        
        for i, text in enumerate(info_texts):
            if text:  # Skip empty strings
                surface = self.font.render(text, True, (0, 0, 0))
                self.screen.blit(surface, (info_x + 10, y_offset + i * 25))
    
    def update_display(self, episode: int, epsilon: float):
        """Ενημέρωση οθόνης"""
        self.screen.fill((255, 255, 255))
        self.draw_zoom_view()
        stats = self.ca_grid.get_statistics()
        self.draw_info_panel(episode, epsilon, stats)
        pygame.display.flip()

def train_cellular_automaton(config: Config) -> CellularAutomatonGrid:
    """Εκπαίδευση του Κυψελιδωτού Αυτομάτου"""
    ca_grid = CellularAutomatonGrid(config)
    ca_grid._initialize_grid()
    ca_grid._setup_neighbors()
    visualizer = CAVisualizer(ca_grid)
    
    epsilon = config.epsilon
    running = True
    
    logger.info(f"Starting training with {config.max_episodes} episodes")
    
    for episode in range(config.max_episodes):
        ca_grid.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < config.max_steps_per_episode and running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
            
            # Choose and execute action
            current_pos = ca_grid.robot_pos
            action = ca_grid.choose_action(current_pos, epsilon)
            next_pos, reward, done = ca_grid.move_robot(action)
            
            # Update CA
            ca_grid.update_cellular_automaton(current_pos, action, reward, next_pos, 
                                            config.alpha, config.gamma)
            
            total_reward += reward
            steps += 1
            
            # Visualization (every 5th episode)
            if episode % 5 == 0 or episode < 10:
                visualizer.update_display(episode, epsilon)
                visualizer.clock.tick(config.fps)
        
        # Episode completed
        ca_grid.episode_rewards.append(total_reward)
        ca_grid.episode_steps.append(steps)
        ca_grid.success_rate.append(1.0 if done else 0.0)
        
        # Epsilon decay
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        
        # Logging
        if episode % 50 == 0:
            stats = ca_grid.get_statistics()
            logger.info(f"Episode {episode}: Reward={total_reward:.2f}, "
                       f"Steps={steps}, Success Rate={stats.get('success_rate', 0):.2f}")
        
        if not running:
            break
    
    pygame.quit()
    return ca_grid

def analyze_results(ca_grid: CellularAutomatonGrid):
    """Ανάλυση και οπτικοποίηση αποτελεσμάτων"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Rewards per episode
    axes[0, 0].plot(ca_grid.episode_rewards)
    axes[0, 0].set_title('Συνολική Ανταμοιβή ανά Επεισόδιο', fontsize=14)
    axes[0, 0].set_xlabel('Επεισόδιο', fontsize=12)
    axes[0, 0].set_ylabel('Συνολική Ανταμοιβή', fontsize=12)
    axes[0, 0].grid(True)

    # 2. Success rate (moving average)
    window_size = 50
    if len(ca_grid.success_rate) >= window_size:
        success_ma = np.convolve(ca_grid.success_rate, np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(success_ma)
    else:
        axes[0, 1].plot(ca_grid.success_rate)
    axes[0, 1].set_title(f'Ποσοστό Επιτυχίας (Κινούμενος Μέσος Όρος, παράθυρο={window_size})', fontsize=14)
    axes[0, 1].set_xlabel('Επεισόδιο', fontsize=12)
    axes[0, 1].set_ylabel('Ποσοστό Επιτυχίας', fontsize=12)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True)

    # 3. Steps per episode
    axes[1, 0].plot(ca_grid.episode_steps)
    axes[1, 0].set_title('Βήματα ανά Επεισόδιο', fontsize=14)
    axes[1, 0].set_xlabel('Επεισόδιο', fontsize=12)
    axes[1, 0].set_ylabel('Βήματα', fontsize=12)
    axes[1, 0].grid(True)

    # 4. Q-values heatmap (βελτιωμένο)
    q_heatmap = np.zeros((ca_grid.size, ca_grid.size))
    visit_heatmap = np.zeros((ca_grid.size, ca_grid.size))

    for i in range(ca_grid.size):
        for j in range(ca_grid.size):
            cell = ca_grid.grid[i][j]
            if cell.state == CellState.OBSTACLE:
                q_heatmap[i, j] = -1
                visit_heatmap[i, j] = -1
            elif cell.state == CellState.GOAL:
                q_heatmap[i, j] = np.max(cell.q_values) + 10
                visit_heatmap[i, j] = cell.visit_count
            else:
                q_heatmap[i, j] = np.max(cell.q_values)
                visit_heatmap[i, j] = cell.visit_count

    q_heatmap_norm = np.copy(q_heatmap)
    q_heatmap_norm[q_heatmap_norm > 0] = (q_heatmap_norm[q_heatmap_norm > 0] -
                                          np.min(q_heatmap_norm[q_heatmap_norm > 0])) / \
                                         (np.max(q_heatmap_norm[q_heatmap_norm > 0]) -
                                          np.min(q_heatmap_norm[q_heatmap_norm > 0]) + 1e-8)

    im1 = axes[1, 1].imshow(q_heatmap_norm, cmap='RdYlBu_r', interpolation='nearest')
    axes[1, 1].set_title('Θερμικός Χάρτης Q-Τιμών (Κανονικοποιημένος)', fontsize=14)
    plt.colorbar(im1, ax=axes[1, 1])

    # 5. Απόδοση ανά run (νέο διάγραμμα)
    if ca_grid.run_performance['successful_episodes']:
        run_data = ca_grid.run_performance
        runs = range(1, len(run_data['successful_episodes']) + 1)

        axes[0, 2].bar(runs, run_data['successful_episodes'], alpha=0.7, color='green')
        axes[0, 2].set_title('Επιτυχημένα Επεισόδια ανά Run', fontsize=14)
        axes[0, 2].set_xlabel('Run', fontsize=12)
        axes[0, 2].set_ylabel('Επιτυχίες', fontsize=12)
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 2].plot(runs, run_data['exploration_efficiency'], 'o-', color='purple')
        axes[1, 2].set_title('Αποδοτικότητα Εξερεύνησης ανά Run', fontsize=14)
        axes[1, 2].set_xlabel('Run', fontsize=12)
        axes[1, 2].set_ylabel('Αποδοτικότητα (%)', fontsize=12)
        axes[1, 2].grid(True)
    else:
        axes[0, 2].text(0.5, 0.5, 'No Run Data Available', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[1, 2].text(0.5, 0.5, 'No Run Data Available', ha='center', va='center', transform=axes[1, 2].transAxes)

    # Περιστροφή ετικετών αν χρειάζεται
    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(15)

    plt.tight_layout(pad=3.0)
    plt.show()

def main():
    """Κύρια συνάρτηση"""
    # Configuration
    config = Config(
        grid_size=30,
        obstacle_density=0.20,
        max_episodes=300,
        epsilon=0.5,
        alpha=0.15
    )
    
    print("🤖 Cellular Automaton Robot Navigation")
    print("=====================================")
    print(f"Grid Size: {config.grid_size}x{config.grid_size}")
    print(f"Obstacle Density: {config.obstacle_density:.1%}")
    print(f"Max Episodes: {config.max_episodes}")
    print("\nPress ESC or close window to stop training early.")
    print("Training starting in 3 seconds...")
    
    time.sleep(3)
    
    # Training
    trained_ca = train_cellular_automaton(config)
    
    # Analysis
    print("\nTraining completed! Analyzing results...")
    analyze_results(trained_ca)
    
    # Final statistics
    final_stats = trained_ca.get_statistics()
    print("\n🏆 Final Results:")
    print("================")
     # Run performance summary
    if trained_ca.run_performance['successful_episodes']:
        print(f"\n📊 Performance per Run (every 50 episodes):")
        print("=" * 45)
        for i, (successes, avg_reward, efficiency) in enumerate(zip(
            trained_ca.run_performance['successful_episodes'],
            trained_ca.run_performance['average_rewards'], 
            trained_ca.run_performance['exploration_efficiency'])):
            print(f"Run {i+1}: {successes} successes, Avg Reward: {avg_reward:.2f}, "
                  f"Exploration: {efficiency:.1f}%")
if __name__ == "__main__":
    main()