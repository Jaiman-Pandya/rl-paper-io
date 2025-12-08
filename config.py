"""Configuration file for hyperparameters and experiment settings."""

# Random seed for reproducibility
# Set this to ensure consistent results across runs
RANDOM_SEED = 42

# Environment settings
ENV_CONFIG = {
    'grid_size': 50,
    'num_opponents': 2,
    'max_steps': 2000
}

# DQN agent hyperparameters
DQN_CONFIG = {
    'lr': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995
}

# Training hyperparameters
TRAIN_CONFIG = {
    'num_episodes': 1000,
    'batch_size': 64,
    'buffer_capacity': 50000,
    'target_update_frequency': 10,  # Update target network every N episodes
    'checkpoint_frequency': 50,  # Save checkpoint every N episodes
    'render': False
}

# Evaluation settings
EVAL_CONFIG = {
    'num_episodes': 100,
    'checkpoint_path': 'checkpoints/dqn_episode_1000.pth',  # Default checkpoint to evaluate
    'render': False
}

# Paths
PATHS = {
    'checkpoints_dir': 'checkpoints',
    'results_dir': 'results'
}

# Reproducibility settings
REPRODUCIBILITY_CONFIG = {
    'random_seed': RANDOM_SEED
}

