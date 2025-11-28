import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def load_training_data(path='results/training_data.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_eval_data(path='results/eval_results.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def smooth_curve(values, window=50):
    if len(values) < window:
        return values

    plot = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        plot.append(np.mean(values[start:end]))

    return plot

def plot_rewards(episodes, rewards, save_path='results/'):
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Rewards')
    smoothed = smooth_curve(rewards, window=50)
    plt.plot(episodes, smoothed, color='darkblue', linewidth=2, label='Smoothed (50 episodes)')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Training Rewards Over Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}rewards.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_scores(episodes, scores, save_path='results/'):
    plt.figure(figsize=(10, 6))

    plt.plot(episodes, scores, alpha=0.3, color='green', label='Raw Scores')

    smoothed = smooth_curve(scores, window=50)
    plt.plot(episodes, smoothed, color='darkgreen', linewidth=2, label='Smoothed (50 episodes)')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Territory Score (%)', fontsize=12)
    plt.title('Territory Capture Over Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}scores.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss(episodes, losses, save_path='results/'):
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, losses, alpha=0.5, color='red', label='Training Loss')

    smoothed = smooth_curve(losses, window=20)
    plt.plot(episodes, smoothed, color='darkred', linewidth=2, label='Smoothed (20 episodes)')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning(episodes, scores, save_path='results/'):
    milestones = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    scores = []

    for episode in milestones:
        if episode <= len(scores):
            start = max(0, episode - 50)
            end = min(len(scores), episode)
            scores.append(np.mean(scores[start:end]))

    plt.figure(figsize=(10, 6))
    plt.plot(milestones[:len(scores)], scores,
             marker='o', linewidth=2, markersize=8, color='purple')

    for i, (episode, score) in enumerate(zip(milestones[:len(scores)], scores)):
        plt.text(episode, score + 0.2, f'{score:.1f}%', ha='center', fontsize=10)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Score (%)', fontsize=12)
    plt.title('Learning Progression at Milestones', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}progression.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_distribution(scores, save_path='results/'):
    final_100 = scores[-100:]

    plt.figure(figsize=(10, 6))
    plt.hist(final_100, bins=20, color='orange', alpha=0.7, edgecolor='black')

    mean_score = np.mean(final_100)
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_score:.2f}%')

    plt.xlabel('Score (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Score Distribution (Final 100 Episodes)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_agent_comparison(results, save_path='results/'):
    agents = list(results.keys())
    mean_scores = [results[agent]['mean_score'] * 100 for agent in agents]
    std_scores = [results[agent]['std_score'] * 100 for agent in agents]

    plt.figure(figsize=(10, 6))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(agents, mean_scores, yerr=std_scores, capsize=10,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    plt.ylabel('Average Score (%)', fontsize=12)
    plt.title('Agent Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, mean_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{score:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_metrics(results, save_path='results/'):
    agents = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    mean_scores = [results[agent]['mean_score'] * 100 for agent in agents]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    axes[0].bar(agents, mean_scores, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Average Score (%)', fontsize=11)
    axes[0].set_title('Mean Performance', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    max_scores = [results[agent]['max_score'] * 100 for agent in agents]
    axes[1].bar(agents, max_scores, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Max Score (%)', fontsize=11)
    axes[1].set_title('Best Performance', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    survivals = [results[agent]['mean_survival'] for agent in agents]
    axes[2].bar(agents, survivals, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Average Steps', fontsize=11)
    axes[2].set_title('Survival Time', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def final_plot(data, save_path='results/'):
    rewards = data['rewards']
    scores = data['scores']
    losses = data['losses']
    episodes = list(range(1, len(rewards) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue')
    smoothed_rewards = smooth_curve(rewards, 50)
    axes[0, 0].plot(episodes, smoothed_rewards, color='darkblue', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, scores, alpha=0.3, color='green')
    smoothed_scores = smooth_curve(scores, 50)
    axes[0, 1].plot(episodes, smoothed_scores, color='darkgreen', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score (%)')
    axes[0, 1].set_title('Territory Capture', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    if losses and len(losses) > 0:
        loss_episodes = list(range(1, len(losses) + 1))
        axes[1, 0].plot(loss_episodes, losses, alpha=0.5, color='red')
        smoothed_losses = smooth_curve(losses, 20)
        axes[1, 0].plot(loss_episodes, smoothed_losses, color='darkred', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

    final_100 = scores[-100:]
    axes[1, 1].hist(final_100, bins=20, color='orange', alpha=0.7, edgecolor='black')
    mean_val = np.mean(final_100)
    axes[1, 1].axvline(mean_val, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Score (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Final Scores (Mean: {mean_val:.1f}%)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_plots():
    training_data = load_training_data()
    eval_data = load_eval_data()

    rewards = training_data['rewards']
    scores = training_data['scores']
    losses = training_data['losses']
    episodes = list(range(1, len(rewards) + 1))

    plot_rewards(episodes, rewards)
    plot_scores(episodes, scores)
    plot_learning(episodes, scores)
    plot_score_distribution(scores)
    plot_agent_comparison(eval_data)
    plot_all_metrics(eval_data)
    final_plot(training_data)

    if losses and len(losses) > 0:
        plot_loss(episodes, losses)

if __name__ == "__main__":
    create_plots()
