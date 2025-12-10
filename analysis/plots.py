"""Generate comparison plots from saved evaluation results.

Loads evaluation results from results/eval_results_latest.json and generates
bar charts comparing agent performance metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_evaluation_results(results_path: str = None) -> dict:
    """Load evaluation results from JSON file.
    
    Args:
        results_path: Path to results file (uses default if None)
        
    Returns:
        Dictionary containing evaluation results
        
    Raises:
        FileNotFoundError: If results file doesn't exist
    """
    if results_path is None:
        results_path = os.path.join(config.PATHS['results_dir'], 'eval_results_latest.json')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Evaluation results file not found: {results_path}\n"
            "Please run evaluation first using training/eval.py or run_experiments.py"
        )
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data

def generate_comparison_plots(results_path: str = None, output_path: str = None):
    """Generate comparison plots from evaluation results.
    
    Creates bar charts comparing mean score and mean reward across agents.
    
    Args:
        results_path: Path to evaluation results JSON (uses default if None)
        output_path: Path to save output plot (uses default if None)
    """
    try:
        data = load_evaluation_results(results_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    results = data['results']
    
    # Extract agent names and metrics (order by agent name for consistency)
    agent_names = sorted(results.keys())
    mean_scores = [results[name]['mean_score'] * 100 for name in agent_names]  # Convert to percentage
    mean_rewards = [results[name]['mean_reward'] for name in agent_names]
    
    # Color mapping for known agents (fallback to default if new agents added)
    color_map = {
        'Random': '#E8505B',
        'Rule-Based': '#F9C74F',
        'DQN': '#4A90E2'
    }
    colors = [color_map.get(name, '#808080') for name in agent_names]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean Score (Territory %)
    bars1 = axes[0].bar(agent_names, mean_scores, color=colors, edgecolor='black', 
                        linewidth=2, alpha=0.85)
    axes[0].set_ylabel('Average Territory (%)', fontsize=13, fontweight='bold')
    axes[0].set_title('Average Performance', fontsize=15, fontweight='bold')
    axes[0].grid(True, alpha=0.25, axis='y', linestyle='--')
    axes[0].set_ylim(0, max(mean_scores) * 1.3 if mean_scores else 10)
    
    for bar, score in zip(bars1, mean_scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + max(mean_scores) * 0.02,
                    f'{score:.1f}%', ha='center', va='bottom',
                    fontsize=13, fontweight='bold')
    
    # Plot 2: Mean Reward
    bars2 = axes[1].bar(agent_names, mean_rewards, color=colors, edgecolor='black',
                        linewidth=2, alpha=0.85)
    axes[1].set_ylabel('Mean Reward', fontsize=13, fontweight='bold')
    axes[1].set_title('Average Reward', fontsize=15, fontweight='bold')
    axes[1].grid(True, alpha=0.25, axis='y', linestyle='--')
    axes[1].set_ylim(0, max(mean_rewards) * 1.2 if mean_rewards else 10)
    
    for bar, reward in zip(bars2, mean_rewards):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + max(mean_rewards) * 0.02,
                    f'{reward:.1f}', ha='center', va='bottom',
                    fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        output_path = os.path.join(config.PATHS['results_dir'], 'agent_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comparison plots saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    generate_comparison_plots()
