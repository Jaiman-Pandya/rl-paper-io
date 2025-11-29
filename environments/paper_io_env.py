import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from environments.logic import PaperIOGame
from environments.renderer import PaperIORenderer

class PaperIOEnv(gym.Env):
    """Gymnasium environment wrapper for Paper.io game.
    
    Implements the standard Gymnasium interface for reinforcement learning.
    The environment provides a 20-dimensional state space and 3 discrete actions.
    """
    env_info = {}
    env_info['render_modes'] = ['human', 'array']
    env_info['render_fps'] = 30

    def __init__(self, grid_size: int = 50, num_opponents: int = 3, mode: Optional[str] = None):
        """Initialize the Paper.io environment.
        
        Args:
            grid_size: Size of the game grid (grid_size x grid_size)
            num_opponents: Number of opponent players
            mode: Rendering mode ('human' for visualization, None for headless)
        """
        super().__init__()
        self.game = PaperIOGame(grid_size=grid_size, num_opponents=num_opponents)
        self.rendering = mode
        self.renderer = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(20,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (initial observation, info dictionary)
        """
        super().reset(seed=seed)

        player = self.game.reset()
        obs = self.game.get_player_state(player)

        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action to take (0=straight, 1=left, 2=right)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        reward, done = self.game.step(action)
        obs = self.game.get_player_state(self.game.players[0])

        ignored = False
        game_info = {
            'score': self.game.players[0].score,
            'alive': self.game.players[0].alive
        }

        return obs, reward, done, ignored, game_info

    def render(self):
        """Render the environment (if rendering mode is enabled)."""
        if self.rendering == 'human':
            if self.renderer is None:
                self.renderer = PaperIORenderer(self.game)
            self.renderer.render()

    def close(self):
        """Clean up rendering resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
