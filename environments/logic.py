import numpy as np
from typing import List, Tuple, Set
from dataclasses import dataclass
from enum import IntEnum

class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Player:
    def __init__(self, x, y, direction, color):
        self.x = x
        self.y = y
        self.direction = direction
        self.color = color
        self.territory = set()
        self.trail = []
        self.alive = True
        self.score = 0.0

    def get_pos(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

class PaperIOGame:
    def __init__(self, grid_size: int = 50, num_opponents: int = 3):
        self.grid_size = grid_size
        self.opponents = num_opponents
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.players: List[Player] = []
        self.steps = 0
        self.maximum_steps = 2000

    def reset(self) -> Player:
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.players = []
        self.steps = 0

        main_player = self.create_player(
            player_id=1,
            x=self.grid_size // 4,
            y=self.grid_size // 4,
            color=(0, 255, 0)
        )
        self.players.append(main_player)

        for i in range(self.opponents):
            opponent = self.create_player(
                player_id=i + 2,
                x = np.random.randint(10, self.grid_size - 10),
                y = np.random.randint(10, self.grid_size - 10),
                color = (np.random.randint(50, 255),
                       np.random.randint(50, 255),
                       np.random.randint(50, 255))
            )

            self.players.append(opponent)

        return self.players[0]

    def create_player(self, player_id: int, x: int, y: int,
                      color: Tuple[int, int, int]) -> Player:

        territory = set()
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x_pos, y_pos = x + dx, y + dy

                if 0 <= x_pos < self.grid_size and 0 <= y_pos < self.grid_size:
                    territory.add((x_pos, y_pos))
                    self.grid[y_pos, x_pos] = player_id

        return Player(
            x = float(x),
            y = float(y),
            direction = Direction.RIGHT,
            color = color,
            territory = territory,
            trail = [],
            alive = True,
            score = len(territory) / (self.grid_size * self.grid_size)
        )

    def step(self, main_player_action: int, opponent_actions: List[int] = None):
        self.steps += 1

        reward = 0
        if self.players[0].alive:
            old_score = self.players[0].score
            dead, gained = self.move(self.players[0], main_player_action)

            if dead:
                reward = -100
                self.players[0].alive = False
            else:
                reward = gained * 200
                reward += 0.1
                if len(self.players[0].trail) > 0:
                    reward -= len(self.players[0].trail) * 0.05

        if opponent_actions is None:
            opponent_actions = [self.ai_player(player) for player in self.players[1:]]

        for i, action in enumerate(opponent_actions):
            if self.players[i + 1].alive:
                self.move(self.players[i + 1], action)

        finished = (not self.players[0].alive or self.steps >= self.maximum_steps or self.players[0].score >= 0.95)

        if self.players[0].score > 0.5:
            reward += 10
        if self.players[0].score > 0.7:
            reward += 20

        return reward, finished

    def move(self, player: Player, action: int) -> Tuple[bool, float]:
        if action == 1:
            player.direction = Direction((player.direction - 1) % 4)
        elif action == 2:
            player.direction = Direction((player.direction + 1) % 4)

        if player.direction == Direction.UP:
            player.y -= 1
        elif player.direction == Direction.DOWN:
            player.y += 1
        elif player.direction == Direction.LEFT:
            player.x -= 1
        elif player.direction == Direction.RIGHT:
            player.x += 1

        if not (0 <= player.x < self.grid_size and 0 <= player.y < self.grid_size):
            return True, 0.0

        pos = player.get_pos()

        if pos in player.territory:
            if len(player.trail) > 0:
                territory_gained = self.gain_grid_pos(player)
                player.trail = []
                return False, territory_gained
            return False, 0.0
        else:
            if pos in player.trail:
                return True, 0.0

            for opponent in self.players:
                if opponent is player:
                    continue
                if pos in opponent.trail:
                    opponent.alive = False
                    opponent.trail = []
                if pos in opponent.territory:
                    pass

            player.trail.append(pos)
            return False, 0.0

    def gain_grid_pos(self, player: Player) -> float:
        if len(player.trail) < 3:
            return 0.0

        old_territory_size = len(player.territory)

        trail_set = set(player.trail)

        for pos in player.trail:
            player.territory.add(pos)
            self.grid[pos[1], pos[0]] = self.players.index(player) + 1

        min_x = min(p[0] for p in trail_set)
        max_x = max(p[0] for p in trail_set)
        min_y = min(p[1] for p in trail_set)
        max_y = max(p[1] for p in trail_set)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if self.inside_trail(x, y, trail_set, player.territory):
                    player.territory.add((x, y))
                    self.grid[y, x] = self.players.index(player) + 1

        new_gained = len(player.territory)
        total_gained = (new_gained - old_territory_size) / (self.grid_size * self.grid_size)
        player.score = new_gained / (self.grid_size * self.grid_size)

        return total_gained

    def inside_trail(self, x: int, y: int, trail: Set[Tuple[int, int]], territory: Set[Tuple[int, int]]) -> bool:
        if (x, y) in trail or (x, y) in territory:
            return True

        crossings = 0
        for tx, ty in trail:
            if ty == y and tx > x:
                crossings += 1

        return crossings % 2 == 1

    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def ai_player(self, player: Player) -> int:
        if not player.alive:
            return 0

        if len(player.trail) > 10:
            if len(player.territory) > 0:
                min_dist = float('inf')
                nearest = None
                player_pos = (player.x, player.y)

                for pos in player.territory:
                    dist = self.manhattan_distance(pos, player_pos)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = pos

                dx = nearest[0] - player.x
                dy = nearest[1] - player.y

                if abs(dx) > abs(dy):
                    target_dir = Direction.RIGHT if dx > 0 else Direction.LEFT
                else:
                    target_dir = Direction.DOWN if dy > 0 else Direction.UP

                diff = (target_dir - player.direction) % 4
                if diff == 1:
                    return 2
                elif diff == 3:
                    return 1

        return np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])

    def get_player_state(self, player: Player) -> np.ndarray:
        if not player.alive:
            return np.zeros(20)

        curr_state = []
        curr_state.append(player.x / self.grid_size)
        curr_state.append(player.y / self.grid_size)
        curr_state.append(player.direction / 4.0)
        curr_state.append(player.score)
        curr_state.append(len(player.trail) / 50.0)

        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (1, -1), (-1, 1), (1, 1)
        ]

        for dx, dy in directions:
            danger = 0.0
            for dist in range(1, 6):
                check_x = int(player.x + dx * dist)
                check_y = int(player.y + dy * dist)

                if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                    danger = 1.0
                    break

                if (check_x, check_y) in player.trail:
                    danger = 1.0 / dist
                    break

                for other in self.players:
                    if other is player:
                        continue
                    if (check_x, check_y) in other.territory or (check_x, check_y) in other.trail:
                        danger = 0.5 / dist
                        break

                if danger > 0:
                    break

            curr_state.append(danger)

        if len(player.trail) > 0 and len(player.territory) > 0:
            min_dist = float('inf')
            nearest = None

            for pos in player.territory:
                dist = abs(pos[0] - player.x) + abs(pos[1] - player.y)
                if dist < min_dist:
                    min_dist = dist
                    nearest = pos

            dist = abs(nearest[0] - player.x) + abs(nearest[1] - player.y)
            curr_state.append(min(dist / 20.0, 1.0))
        else:
            curr_state.append(0.0)

        curr_state.append(1.0 if player.get_pos() in player.territory else 0.0)
        num_alive_opponents = sum(1 for p in self.players[1:] if p.alive)
        curr_state.append(num_alive_opponents / self.opponents)
        curr_state.append(self.steps / self.maximum_steps)
        nearest_opponent_dist = float('inf')

        for other in self.players[1:]:
            if other.alive:
                dist = abs(other.x - player.x) + abs(other.y - player.y)
                nearest_opponent_dist = min(nearest_opponent_dist, dist)

        if nearest_opponent_dist != float('inf'):
            curr_state.append(nearest_opponent_dist / nearest_opponent_dist)
        else:
            curr_state.append(1.0)

        curr_state.append(len(player.territory) / (self.grid_size * self.grid_size))

        max_territory = 0
        for p in self.players:
            if p.alive:
                territory_size = len(p.territory)
                if territory_size > max_territory:
                    max_territory = territory_size

        curr_state.append(max_territory / (self.grid_size * self.grid_size))

        return np.array(curr_state, dtype=np.float32)
