# Reinforcement Learning for Paper.io

A reinforcement learning project implementing Deep Q-Network (DQN) to train an agent to play Paper.io, a territory-capture game.

![Paper.io Gameplay](https://github.com/user-attachments/assets/245d46a2-dfd2-4a9a-9c81-df5b34c2868b)

## Project Overview

This project implements a DQN agent to learn optimal strategies for Paper.io, where the goal is to capture territory by drawing trails and returning to your own territory. The agent competes against rule-based opponents in a grid-based environment.

### Problem Statement

Paper.io is a real-time strategy game where players:
- Move around a grid to draw trails
- Must return to their territory to capture the enclosed area
- Avoid colliding with their own trail or opponents
- Compete to capture the most territory

### AI Techniques Used

- **Deep Q-Network (DQN)**: Deep reinforcement learning algorithm with:
  - Experience replay buffer
  - Target network for stable Q-learning
  - Epsilon-greedy exploration
  - Feedforward neural network (128-128-64 architecture)

- **Baseline Methods**:
  - Random agent: Uniform random action selection
  - Rule-based agent: Hand-crafted heuristics based on trail length and danger detection

## Project Structure

```
rl-paper-io/
├── agents/             # Agent implementations
│   ├── dqn.py          # DQN agent with neural network
│   ├── random.py       # Random baseline agent
│   └── rule.py         # Rule-based baseline agent
├── environments/       # Game environment
│   ├── logic.py        # Core game logic and state management
│   ├── paper_io_env.py # Gymnasium environment wrapper
│   └── renderer.py     # Pygame visualization
├── training/           # Training and evaluation scripts
│   ├── train.py        # DQN training script
│   └── eval.py         # Evaluation and comparison script
├── utils/              # Utility modules
│   └── buffer.py       # Experience replay buffer
├── analysis/           # Analysis and visualization
│   └── plots.py        # Comparison plots
├── checkpoints/        # Saved model checkpoints
├── results/            # Evaluation results and plots
├── config.py           # Configuration and hyperparameters
└── run_experiments.py  # Main experiment runner
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rl-paper-io
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a DQN agent:
```bash
python run_experiments.py --mode train
```

Or use the training script directly:
```bash
python training/train.py
```

### Evaluation

Evaluate and compare all agents:
```bash
python run_experiments.py --mode eval --checkpoint checkpoints/dqn_episode_1000.pth
```

Or use the evaluation script directly:
```bash
python training/eval.py
```

### Full Pipeline

Run both training and evaluation:
```bash
python run_experiments.py --mode both
```

### Configuration

Hyperparameters and settings can be modified in `config.py`:
- Environment settings (grid size, opponents)
- DQN hyperparameters (learning rate, gamma, epsilon)
- Training settings (episodes, batch size, buffer capacity)
- Evaluation settings

## Results

The project includes:
- Training curves showing learning progress (rewards, scores, losses)
- Agent comparison metrics (mean score, max score, survival time)
- Visualization of agent performance

### Expected Performance

- **Random Agent**: ~0.4% average territory
- **Rule-Based Agent**: ~0.4% average territory  
- **DQN Agent**: ~4-12% average territory (after training)

## Code Quality

- **Modularity**: Clear separation of concerns (agents, environment, training, evaluation)
- **Documentation**: Docstrings for all classes and functions
- **Type Hints**: Type annotations throughout the codebase
- **Configuration**: Centralized hyperparameter management
- **Baselines**: Multiple baseline methods for comparison

## Implementation Details

### State Representation

20-dimensional state vector including:
- Player position and direction
- Territory score and trail length
- Danger detection in 8 directions
- Distance to nearest territory
- Opponent information

### Action Space

3 discrete actions:
- 0: Continue straight
- 1: Turn left
- 2: Turn right

### Reward Function

- +200 × territory gained
- +0.1 per step (survival bonus)
- -0.05 × trail length (penalty for long trails)
- +10 for >50% territory
- +20 for >70% territory
- -100 for death

## Future Improvements

- [ ] Implement Double DQN or Dueling DQN variants
- [ ] Add prioritized experience replay
- [ ] Experiment with different network architectures
- [ ] Implement curriculum learning
- [ ] Add more sophisticated baseline agents
- [ ] Improve reward shaping
- [ ] Add hyperparameter tuning framework
- [ ] Implement multi-agent training

