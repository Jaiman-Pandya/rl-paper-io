import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

class DQN(nn.Module):
    """Deep Q-Network (DQN) neural network architecture.
    
    A feedforward neural network that maps state observations to Q-values
    for each possible action.
    """
    def __init__(self, state_dimensions: int, action_dimensions: int):
        """Initialize the DQN network.
        
        Args:
            state_dimensions: Size of the state observation vector
            action_dimensions: Number of possible actions
        """
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dimensions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dimensions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(x)

class DQNAgent:
    """Deep Q-Network (DQN) agent using double Q-learning with target network.
    
    Implements the DQN algorithm with epsilon-greedy exploration, experience replay,
    and a separate target network for stable Q-value estimation.
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001, 
                 gamma: float = 0.99, epsilon_start: float = 1.0, 
                 epsilon_end: float = 0.01, epsilon_decay: float = 0.995):
        """Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions
            lr: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
        """
        self.device = torch.device("cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.policy = DQN(state_dim, action_dim).to(self.device)
        self.target = DQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether the agent is in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy(state)

            return q_values.argmax().item()

    def train(self, batch: Tuple) -> float:
        """Train the agent on a batch of experiences.
        
        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
            
        Returns:
            Training loss value
        """
        states, actions, rewards, adj_states, dones = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        adj_states = torch.FloatTensor(adj_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        curr = self.policy(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target(adj_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss(curr.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Update the target network with the current policy network weights."""
        self.target.load_state_dict(self.policy.state_dict())

    def decay_rate(self):
        """Decay the exploration rate epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """Save the agent's state to a file.
        
        Args:
            path: Path to save the checkpoint
        """
        torch.save({
            'policy': self.policy.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str):
        """Load the agent's state from a file.
        
        Args:
            path: Path to load the checkpoint from
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
