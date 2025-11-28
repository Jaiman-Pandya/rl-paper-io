import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from environments.paper_io_env import PaperIOEnv
from agents.dqn import DQNAgent
from agents.random import RandomAgent
from agents.rule import RuleBasedAgent

def eval(agent, num_episodes=100, render=False):
    if render:
        render_mode = 'human'
    else:
        render_mode = None

    env = PaperIOEnv(grid_size = 50, num_opponents = 2, mode= render_mode)

    scores = []
    survival_times = []

    for episode in range(num_episodes):
        state, i = env.reset()
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            steps += 1

            if render:
                env.render()

        scores.append(info['score'])
        survival_times.append(steps)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Score = {info['score']:.1%}, Steps = {steps}")

    env.close()

    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'mean_survival': np.mean(survival_times),
        'scores': scores
    }

def compare():
    print("We now compare the agent performances!")

    state_dimensions = 20
    action_dimensions = 3

    dqn = DQNAgent(state_dimensions, action_dimensions)
    dqn.load('checkpoints/dqn_episode_1000.pth')
    dqn.epsilon = 0.0

    random = RandomAgent(action_dimensions)
    rule_based = RuleBasedAgent(action_dimensions)

    agents = {
        'DQN': dqn,
        'Random': random,
        'Rule-Based': rule_based
    }

    results = {}

    for name, agent in agents.items():
        print(f"\nEvaluating {name} agent...")
        results[name] = eval(agent, num_episodes=100)

        print(f"{name} Results:")
        print(f"Mean Score: {results[name]['mean_score']:.1%} ± {results[name]['std_score']:.1%}")
        print(f"Max Score: {results[name]['max_score']:.1%}")
        print(f"Mean Survival: {results[name]['mean_survival']:.0f} steps")

    return results

if __name__ == "__main__":
    results = compare()
