import numpy as np
import random
from collections import deque
from typing import Tuple

class ReplayBuffer:
    """Experience replay buffer for DQN training.
    
    Stores transitions (state, action, reward, next_state, done) and
    samples random batches for training to break correlation in sequential data.
    """
    def __init__(self, capacity: int = 10000):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
