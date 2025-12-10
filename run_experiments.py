"""Main script to run training and evaluation experiments.

This script provides a unified interface to train and evaluate agents.
"""

import os
import argparse
from training.train import train_dqn
from training.eval import compare
import config

def main():
    parser = argparse.ArgumentParser(description='Run Paper.io RL experiments')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'],
                        default='both', help='Mode: train, eval, or both')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes (overrides config)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training/evaluation')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(config.PATHS['checkpoints_dir'], exist_ok=True)
    os.makedirs(config.PATHS['results_dir'], exist_ok=True)
    
    if args.mode in ['train', 'both']:
        print("=" * 60)
        print("TRAINING MODE")
        print("=" * 60)
        num_episodes = args.episodes if args.episodes else config.TRAIN_CONFIG['num_episodes']
        render = args.render if args.render else config.TRAIN_CONFIG['render']
        agent, rewards, scores, run_dir = train_dqn(num_episodes=num_episodes, render=render)
        print(f"\nTraining completed! Run directory: {run_dir}")
    
    if args.mode in ['eval', 'both']:
        print("\n" + "=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)
        checkpoint_path = args.checkpoint if args.checkpoint else config.EVAL_CONFIG['checkpoint_path']
        results = compare(checkpoint_path=checkpoint_path)
        print("\nEvaluation completed!")
        
        # TODO: Save results to file for further analysis
        # TODO: Generate comparison plots automatically

if __name__ == "__main__":
    main()

