import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, Any
import random
import torch
from environments.paper_io_env import PaperIOEnv
from agents.dqn import DQNAgent
from agents.random import RandomAgent
from agents.rule import RuleBasedAgent
import config

def set_random_seeds(seed: int):
    """Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value to use for all generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set CUDA seeds if available (for GPU reproducibility)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval(agent: Any, num_episodes: int = None, render: bool = None) -> Dict[str, Any]:
    """Evaluate an agent on the Paper.io environment.
    
    Args:
        agent: Agent to evaluate (must have select_action method)
        num_episodes: Number of episodes to run (uses config default if None)
        render: Whether to render the environment (uses config default if None)
        
    Returns:
        Dictionary with evaluation metrics:
        - mean_score: Average territory percentage
        - std_score: Standard deviation of scores
        - max_score: Maximum score achieved
        - mean_survival: Average survival time in steps
        - scores: List of all episode scores
    """
    if num_episodes is None:
        num_episodes = config.EVAL_CONFIG['num_episodes']
    if render is None:
        render = config.EVAL_CONFIG['render']
    
    # Set random seeds for reproducibility
    set_random_seeds(config.RANDOM_SEED)
    
    if render:
        render_mode = 'human'
    else:
        render_mode = None

    env = PaperIOEnv(
        grid_size=config.ENV_CONFIG['grid_size'],
        num_opponents=config.ENV_CONFIG['num_opponents'],
        mode=render_mode
    )

    scores = []
    survival_times = []

    for episode in range(num_episodes):
        # Reset with seed for reproducibility (each episode gets a deterministic seed)
        state, _ = env.reset(seed=config.RANDOM_SEED + episode)
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

def compare(checkpoint_path: str = None) -> Dict[str, Dict[str, Any]]:
    """Compare performance of DQN, Random, and Rule-Based agents.
    
    Args:
        checkpoint_path: Path to DQN checkpoint (uses config default if None)
        
    Returns:
        Dictionary mapping agent names to their evaluation results
    """
    # Set random seeds for reproducibility
    set_random_seeds(config.RANDOM_SEED)
    
    print("Comparing agent performances...")

    if checkpoint_path is None:
        checkpoint_path = config.EVAL_CONFIG['checkpoint_path']
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found. "
              "Please train a model first or specify a valid checkpoint path.")
        print("Skipping DQN evaluation, only evaluating baselines.")
        dqn = None
    else:
        state_dimensions = 20
        action_dimensions = 3
        dqn = DQNAgent(state_dimensions, action_dimensions)
        dqn.load(checkpoint_path)
        dqn.epsilon = 0.0  # Disable exploration during evaluation

    random = RandomAgent(3)
    rule_based = RuleBasedAgent(3)

    agents = {}
    if dqn is not None:
        agents['DQN'] = dqn
    agents['Random'] = random
    agents['Rule-Based'] = rule_based

    results = {}

    for name, agent in agents.items():
        print(f"\nEvaluating {name} agent...")
        results[name] = eval(agent, num_episodes=config.EVAL_CONFIG['num_episodes'])

        print(f"{name} Results:")
        print(f"  Mean Score: {results[name]['mean_score']:.1%} ± {results[name]['std_score']:.1%}")
        print(f"  Max Score: {results[name]['max_score']:.1%}")
        print(f"  Mean Survival: {results[name]['mean_survival']:.0f} steps")

    return results

if __name__ == "__main__":
    results = compare()
