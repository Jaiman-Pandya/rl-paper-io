import numpy as np

class RuleBasedAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state, training=True):
        length = state[4]
        inside_territory = state[14]

        if length > 0.2:
            return np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])

        danger_state = state[5]

        if danger_state > 0.5:
            return np.random.choice([1, 2])

        return np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
