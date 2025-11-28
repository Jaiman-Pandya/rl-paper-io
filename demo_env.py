from agents.random import RandomAgent
from environments.paper_io_env import PaperIOEnv

env = PaperIOEnv(grid_size=30, num_opponents=2, mode='human')
agent = RandomAgent(action_dim=3)

for episode in range(5):
    state, i = env.reset()
    done = False
    steps = 0

    while not done and steps < 500:
        action = agent.select_action(state)
        state, reward, done, truncated, info = env.step(action)
        env.render()
        steps += 1

    print(f"Episode {episode + 1}: Score = {info['score']}, Steps = {steps}")

env.close()
