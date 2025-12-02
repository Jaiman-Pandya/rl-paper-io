import numpy as np
from typing import Optional

class RuleBasedAgent:
    """Rule-based baseline agent with simple heuristics.
    
    Uses hand-crafted rules based on trail length, danger detection,
    and territory status to select actions. Used as a stronger baseline
    than random agent.
    This agent does not learn, it only applies a simple "defensive"
    strategy, randomly avoiding obstables, and prefering to return to it's
    territory after a certain trail length. 
    Other possible Rule based agents could implement riskier strategies,
    such as going towards dangers, longer trails, or going in one 
    particular direction.
    """
    def __init__(self, action_dim: int):
        """Initialize the rule-based agent.
        
        Args:
            action_dim: Number of possible actions
        """
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action based on hand-crafted rules.
        
        Rules:
        - If trail is long (>0.2), prefer turning to return to territory by making more frequent turns.
        - If danger detected, turn to avoid collision
        - Otherwise, prefer going straight
        
        Args:
            state: Current state observation (20-dimensional)
            training: Whether in training mode (ignored)
            
        Returns:
            Selected action index (0=straight, 1=left, 2=right)
        """
        length = state[4]  # Normalized trail length
        inside_territory = state[14]  # Whether inside own territory

        # If trail is long, prefer turning to return to territory by making more frequent turns
        if length > 0.2:
            # Probabilities: 50% straight, 25% left, 25% right
            return np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])

        danger_state = state[5]  # Danger in forward direction

        # If danger detected, turn to avoid
        if danger_state > 0.5:
            # Probabilities: Random left OR right
            return np.random.choice([1, 2])

        # Default: prefer going straight (agent wants to travel continually more than changing directions) 
        # Probabilities: 80% straight, 10% left, 10% right
        return np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
