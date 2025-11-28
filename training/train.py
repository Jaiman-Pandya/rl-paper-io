import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from environments.paper_io_env import PaperIOEnv
from agents.dqn import DQNAgent
from utils.buffer import ReplayBuffer

def train_dqn(num_episodes=1000, render=False):
    if render:
        mode = 'human'
    else:
        mode = None

    env = PaperIOEnv(grid_size=50, num_opponents=2, mode=mode)

    state_dimensions = env.observation_space.shape[0]
    action_dimensions = env.action_space.n
    agent = DQNAgent(state_dimensions, action_dimensions)

    buffer = ReplayBuffer(capacity=50000)

    batch_size = 64
    update_time = 10

    rewards = []
    scores = []
    losses = []

    print("Beginning the training process!")

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []

        done = False
        while not done:
            action = agent.select_action(state)

            adj_state, reward, done, complete, info = env.step(action)
            done = done or complete

            buffer.push(state, action, reward, adj_state, done)

            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                loss = agent.train(batch)
                episode_loss.append(loss)

            episode_reward += reward
            state = adj_state

            if render:
                env.render()

        if episode % update_time == 0:
            agent.update_target()
        agent.decay_rate()

        rewards.append(episode_reward)
        scores.append(info['score'])

        if episode_loss:
            losses.append(np.mean(episode_loss))

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards[-50:])
            avg_score = np.mean(scores[-50:])
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Score: {avg_score:.1%}")
            print(f"Epsilon: {agent.epsilon:.3f}")

            agent.save(f'checkpoints/dqn_episode_{episode + 1}.pth')

    env.close()
    plot_training_results(rewards, scores, losses)

    return agent, rewards, scores

def plot_training_results(rewards, scores, losses):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True)

    axes[1].plot(scores)
    axes[1].set_title('Episode Scores (Territory %)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score')
    axes[1].grid(True)

    if losses:
        axes[2].plot(losses)
        axes[2].set_title('Training Loss')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    agent, rewards, scores = train_dqn(num_episodes=1000, render=True)
