import gymnasium as gym
import numpy as np
from gymnasium import spaces
from environments.logic import PaperIOGame
from environments.renderer import PaperIORenderer

class PaperIOEnv(gym.Env):
    env_info = {}
    env_info['render_modes'] = ['human', 'array']
    env_info['render_fps'] = 30

    def __init__(self, grid_size=50, num_opponents=3, mode=None):
        super().__init__()
        self.game = PaperIOGame(grid_size=grid_size, num_opponents=num_opponents)
        self.rendering = mode
        self.renderer = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(20), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        player = self.game.reset()
        obs = self.game.get_player_state(player)

        return obs, {}

    def step(self, action):
        reward, done = self.game.step(action)
        obs = self.game.get_player_state(self.game.players[0])

        ignored = False
        game_info = {
            'score': self.game.players[0].score,
            'alive': self.game.players[0].alive
        }

        return obs, reward, done, ignored, game_info

    def render(self):
        if self.rendering == 'human':
            if self.renderer is None:
                self.renderer = PaperIORenderer(self.game)
            self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
