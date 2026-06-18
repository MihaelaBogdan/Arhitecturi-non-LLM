import random
import math
from enum import Enum
from collections import namedtuple
import numpy as np

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20

class SnakeGameHeadless:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.prev_dist = self._get_dist_to_food()

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _get_dist_to_food(self):
        return math.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)

    def play_step(self, action):
        self.frame_iteration += 1
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -20
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 20
            self._place_food()
            self.prev_dist = self._get_dist_to_food()
        else:
            self.snake.pop()
            curr_dist = self._get_dist_to_food()
            if curr_dist < self.prev_dist:
                reward = 0.1
            else:
                reward = -0.15
            self.prev_dist = curr_dist
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)


def get_astar_path(snake_body, head, food, w=400, h=400, block_size=20):
    cols = int(w // block_size)
    rows = int(h // block_size)
    
    start = (int(head.x // block_size), int(head.y // block_size))
    end = (int(food.x // block_size), int(food.y // block_size))
    
    obstacles = set((int(pt.x // block_size), int(pt.y // block_size)) for pt in snake_body)
    obstacles.discard(start)  # Head itself is the start, remove it from obstacles
    
    open_set = {start}
    came_from = {}
    
    g_score = {start: 0}
    f_score = {start: abs(start[0] - end[0]) + abs(start[1] - end[1])}
    
    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        
        if current == end:
            path = []
            while current in came_from:
                path.append(Point(current[0] * block_size, current[1] * block_size))
                current = came_from[current]
            path.reverse()
            return path
            
        open_set.remove(current)
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if neighbor[0] < 0 or neighbor[0] >= cols or neighbor[1] < 0 or neighbor[1] >= rows:
                continue
            if neighbor in obstacles:
                continue
                
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                open_set.add(neighbor)
                
    return None


class MultiAgentSnakeGame:
    def __init__(self, w=400, h=400):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        # Snake 1: DQN (Blue) - Starts left, moves right
        self.direction_dqn = Direction.RIGHT
        self.head_dqn = Point(100, 200)
        self.snake_dqn = [
            self.head_dqn,
            Point(self.head_dqn.x - BLOCK_SIZE, self.head_dqn.y),
            Point(self.head_dqn.x - 2 * BLOCK_SIZE, self.head_dqn.y)
        ]
        self.score_dqn = 0
        self.dead_dqn = False
        self.frame_iteration_dqn = 0
        
        # Snake 2: Tree (Yellow) - Starts right, moves left
        self.direction_tree = Direction.LEFT
        self.head_tree = Point(280, 200)
        self.snake_tree = [
            self.head_tree,
            Point(self.head_tree.x + BLOCK_SIZE, self.head_tree.y),
            Point(self.head_tree.x + 2 * BLOCK_SIZE, self.head_tree.y)
        ]
        self.score_tree = 0
        self.dead_tree = False
        self.frame_iteration_tree = 0

        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake_dqn or self.food in self.snake_tree:
            self._place_food()

    def is_collision(self, pt, snake_own, snake_other, is_dqn=True):
        # Boundary check
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check hitting own body (excluding head)
        if pt in snake_own[1:]:
            return True
        # Check hitting other snake's body (if other snake is active/alive)
        other_active = not self.dead_tree if is_dqn else not self.dead_dqn
        if other_active and pt in snake_other:
            return True
        return False

    def play_step(self, action_dqn, action_tree):
        # 1. Move DQN Snake
        if not self.dead_dqn:
            self.frame_iteration_dqn += 1
            self._move(action_dqn, is_dqn=True)
            self.snake_dqn.insert(0, self.head_dqn)

        # 2. Move Tree Snake
        if not self.dead_tree:
            self.frame_iteration_tree += 1
            self._move(action_tree, is_dqn=False)
            self.snake_tree.insert(0, self.head_tree)

        # 3. Check food consumption
        eaten_dqn = False
        eaten_tree = False
        
        if not self.dead_dqn and self.head_dqn == self.food:
            eaten_dqn = True
        if not self.dead_tree and self.head_tree == self.food:
            eaten_tree = True

        if eaten_dqn or eaten_tree:
            if eaten_dqn:
                self.score_dqn += 1
                self.frame_iteration_dqn = 0
            if eaten_tree:
                self.score_tree += 1
                self.frame_iteration_tree = 0
            self._place_food()
        
        # Pop tails if not eating
        if not self.dead_dqn:
            if not eaten_dqn:
                self.snake_dqn.pop()
        if not self.dead_tree:
            if not eaten_tree:
                self.snake_tree.pop()

        # 4. Check collisions and updates
        if not self.dead_dqn:
            if self.is_collision(self.head_dqn, self.snake_dqn, self.snake_tree, is_dqn=True) or \
               self.frame_iteration_dqn > 100 * len(self.snake_dqn):
                self.dead_dqn = True
                
        if not self.dead_tree:
            if self.is_collision(self.head_tree, self.snake_tree, self.snake_dqn, is_dqn=False) or \
               self.frame_iteration_tree > 100 * len(self.snake_tree):
                self.dead_tree = True

        done = self.dead_dqn or self.dead_tree
        return done, self.score_dqn, self.score_tree

    def _move(self, action, is_dqn=True):
        direction = self.direction_dqn if is_dqn else self.direction_tree
        head = self.head_dqn if is_dqn else self.head_tree
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        if is_dqn:
            self.direction_dqn = new_dir
        else:
            self.direction_tree = new_dir
            
        x = head.x
        y = head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE
            
        if is_dqn:
            self.head_dqn = Point(x, y)
        else:
            self.head_tree = Point(x, y)

