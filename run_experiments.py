"""Main script to run training and evaluation experiments.

This script provides a unified interface to train and evaluate agents.
It orchestrates the complete pipeline:

1. Training: Trains a DQN agent and saves metrics/logs to results/runs/<timestamp>/
2. Evaluation: Evaluates all agents (DQN, Random, Rule-Based) and saves results
3. Visualization: Automatically generates comparison plots from evaluation results

All results are saved to disk for later analysis and reproducibility.
"""

import os
import argparse
from training.train import train_dqn
from training.eval import compare
from analysis.plots import generate_comparison_plots
import config

def main():
    """Main entry point for running training and evaluation pipeline.
    
    Orchestrates the complete experiment pipeline:
    1. Training (optional): Trains DQN agent, saves config and metrics to run directory
    2. Evaluation: Evaluates all agents, saves results to JSON/CSV
    3. Visualization: Generates comparison plots from evaluation results
    
    All outputs are automatically saved to disk for reproducibility.
    """
    parser = argparse.ArgumentParser(description='Run Paper.io RL experiments')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'],
                        default='both', help='Mode: train, eval, or both')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes (overrides config)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training/evaluation')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating comparison plots after evaluation')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(config.PATHS['checkpoints_dir'], exist_ok=True)
    os.makedirs(config.PATHS['results_dir'], exist_ok=True)
    
    run_id = None  # Track training run ID to link with evaluation
    
    if args.mode in ['train', 'both']:
        print("=" * 60)
        print("TRAINING MODE")
        print("=" * 60)
        num_episodes = args.episodes if args.episodes else config.TRAIN_CONFIG['num_episodes']
        render = args.render if args.render else config.TRAIN_CONFIG['render']
        agent, rewards, scores, run_dir = train_dqn(num_episodes=num_episodes, render=render)
        print(f"\nTraining completed! Run directory: {run_dir}")
        
        # Extract run ID from directory name (timestamp)
        run_id = os.path.basename(run_dir)
        
        # Update checkpoint path to use the latest checkpoint from this training run
        if args.checkpoint is None:
            # Use the final checkpoint from this training run
            final_checkpoint = os.path.join(config.PATHS['checkpoints_dir'], 
                                          f'dqn_episode_{num_episodes}.pth')
            if os.path.exists(final_checkpoint):
                args.checkpoint = final_checkpoint
    
    if args.mode in ['eval', 'both']:
        print("\n" + "=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)
        checkpoint_path = args.checkpoint if args.checkpoint else config.EVAL_CONFIG['checkpoint_path']
        
        # Run evaluation with run_id to link back to training run
        results = compare(checkpoint_path=checkpoint_path, run_id=run_id)
        print("\nEvaluation completed!")
        
        # Automatically generate comparison plots from saved results
        if not args.no_plots:
            print("\n" + "=" * 60)
            print("GENERATING PLOTS")
            print("=" * 60)
            try:
                generate_comparison_plots()
                print("Plots generated successfully!")
            except FileNotFoundError as e:
                print(f"Warning: Could not generate plots - {e}")
                print("This is expected if evaluation results file doesn't exist yet.")
            except Exception as e:
                print(f"Error generating plots: {e}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results directory: {config.PATHS['results_dir']}")
    if run_id:
        print(f"Training run ID: {run_id}")
    print("Check results/ directory for:")
    print("  - eval_results_latest.json (full evaluation data)")
    print("  - eval_results_latest.csv (summary table)")
    print("  - agent_comparison.png (comparison plots)")
    if run_id:
        print(f"  - runs/{run_id}/ (training run data)")

if __name__ == "__main__":
    main()

