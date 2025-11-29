import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List

from environments.paper_io_env import PaperIOEnv
from agents.dqn import DQNAgent
from utils.buffer import ReplayBuffer
import config

def train_dqn(num_episodes: int = None, render: bool = None) -> Tuple[DQNAgent, List[float], List[float]]:
    """Train a DQN agent on the Paper.io environment.
    
    Args:
        num_episodes: Number of training episodes (uses config default if None)
        render: Whether to render during training (uses config default if None)
        
    Returns:
        Tuple of (trained agent, list of episode rewards, list of episode scores)
    """
    if num_episodes is None:
        num_episodes = config.TRAIN_CONFIG['num_episodes']
    if render is None:
        render = config.TRAIN_CONFIG['render']
        
    if render:
        mode = 'human'
    else:
        mode = None

    env = PaperIOEnv(
        grid_size=config.ENV_CONFIG['grid_size'],
        num_opponents=config.ENV_CONFIG['num_opponents'],
        mode=mode
    )

    state_dimensions = env.observation_space.shape[0]
    action_dimensions = env.action_space.n
    agent = DQNAgent(
        state_dimensions, 
        action_dimensions,
        lr=config.DQN_CONFIG['lr'],
        gamma=config.DQN_CONFIG['gamma'],
        epsilon_start=config.DQN_CONFIG['epsilon_start'],
        epsilon_end=config.DQN_CONFIG['epsilon_end'],
        epsilon_decay=config.DQN_CONFIG['epsilon_decay']
    )

    buffer = ReplayBuffer(capacity=config.TRAIN_CONFIG['buffer_capacity'])

    batch_size = config.TRAIN_CONFIG['batch_size']
    update_time = config.TRAIN_CONFIG['target_update_frequency']
    checkpoint_freq = config.TRAIN_CONFIG['checkpoint_frequency']

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

        if (episode + 1) % checkpoint_freq == 0:
            avg_reward = np.mean(rewards[-checkpoint_freq:])
            avg_score = np.mean(scores[-checkpoint_freq:])
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Score: {avg_score:.1%}")
            print(f"Epsilon: {agent.epsilon:.3f}")

            checkpoint_path = os.path.join(
                config.PATHS['checkpoints_dir'],
                f'dqn_episode_{episode + 1}.pth'
            )
            agent.save(checkpoint_path)

    env.close()
    plot_training_results(rewards, scores, losses)

    return agent, rewards, scores

def plot_training_results(rewards: List[float], scores: List[float], losses: List[float]):
    """Plot training curves for rewards, scores, and losses.
    
    Args:
        rewards: List of episode rewards
        scores: List of episode scores (territory percentages)
        losses: List of training losses
    """
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
    os.makedirs(config.PATHS['checkpoints_dir'], exist_ok=True)
    agent, rewards, scores = train_dqn()
