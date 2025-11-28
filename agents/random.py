import numpy as np

class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state, training=True):
        return np.random.randint(0, self.action_dim)
