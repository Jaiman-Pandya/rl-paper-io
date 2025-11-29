import numpy as np
from typing import Optional

class RandomAgent:
    """Baseline agent that selects actions uniformly at random.
    
    Used as a simple baseline for comparison with learned agents.
    """
    def __init__(self, action_dim: int):
        """Initialize the random agent.
        
        Args:
            action_dim: Number of possible actions
        """
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select a random action.
        
        Args:
            state: Current state (ignored for random agent)
            training: Whether in training mode (ignored)
            
        Returns:
            Random action index
        """
        return np.random.randint(0, self.action_dim)
