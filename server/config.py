# config.py
import random
import numpy as np
import torch

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Game constants
ACTIONS = ['move_up', 'move_down', 'move_left', 'move_right', 'drop_trap', 'explode_traps', 'wait']

# Bot configuration
USE_PLAYER_MODEL = False  # Set to True to use player-trained models for some bots
PLAYER_MODEL_BOTS = [0, 1]  # Which bot indices should use the player model

# Neural network configuration
# Feature breakdown:
# - Position: 2
# - Direction: 2 
# - Bot stats: 3
# - Game progress: 2
# - Visible treasures: 10 (5 treasures × 2 coordinates)
# - Visible bots: 12 (3 bots × 4 features)
# - Visible traps: 9 (3 traps × 3 features)
# - Obstacle grid: 25 (5×5 grid)
# - Bot scores: 3
# - Explosion available: 1
# Total: 69 features

STATE_SIZE = 69  # Updated to match actual feature count
ACTION_SIZE = len(ACTIONS)

# Training hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 32
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000

# Server configuration
HOST = '0.0.0.0'
PORT = 5001