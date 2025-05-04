# agent_config.py
"""
Agent Configuration System

This module provides configuration for different agent types in the game.
You can easily configure which bots use which type of agent.
"""

# Agent Types
AGENT_TYPES = {
    'PLAYER_MODEL': 'player',     # LSTM-DQN trained on human data
    'LSTM_DQN': 'lstm_dqn',      # LSTM-DQN with RL training
    'STANDARD_DQN': 'standard',   # Standard DQN
    'MAIN_AGENT': 'main'         # Uses the main agent (either LSTM-DQN or standard)
}

# Main agent configuration
USE_LSTM_DQN_FOR_MAIN = True  # If True, main agent uses LSTM-DQN; if False, uses standard DQN

# Bot configurations - specify which agent type each bot should use
BOT_CONFIGS = {
    # Bot ID: Agent Type
    '0': AGENT_TYPES['PLAYER_MODEL'],    # Bot 0 uses player model
    '1': AGENT_TYPES['LSTM_DQN'],        # Bot 1 uses LSTM-DQN
    '2': AGENT_TYPES['LSTM_DQN'],        # Bot 2 uses LSTM-DQN
    '3': AGENT_TYPES['STANDARD_DQN'],    # Bot 3 uses standard DQN
    # Add more bot configurations as needed
}

# Model paths
MODEL_PATHS = {
    'player_model': 'lstm_dqn_player_model.pth',
    'lstm_dqn': 'lstm_dqn_model.pth',
    'standard_dqn': 'standard_dqn_model.pth',
    'main_agent': 'main_agent_model.pth'
}

# LSTM-DQN specific configurations
LSTM_CONFIG = {
    'hidden_size': 128,
    'lstm_layers': 2,
    'sequence_length': 10,
    'dropout_rate': 0.2
}

# Training configurations for different agent types
TRAINING_CONFIG = {
    'player_model': {
        'pretrain_epochs': 10,
        'fine_tune_episodes': 100,
        'batch_size': 64,
        'learning_rate': 0.001
    },
    'lstm_dqn': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'target_update': 10,
        'memory_capacity': 10000
    },
    'standard_dqn': {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'target_update': 100,
        'memory_capacity': 10000
    }
}

# Reward shaping configurations
REWARD_CONFIG = {
    'treasure_collection': 50,
    'spot_enemy': 5,
    'spot_enemy_with_treasure': 3,
    'enemy_elimination': 100,
    'eliminate_enemy_with_treasure_multiplier': 20,
    'wasted_explosion_penalty': -30,
    'being_spotted_penalty': -5,
    'death_penalty': -100,
    'movement_cost': -0.1,
    'wait_penalty': -2,
    'strategic_trap_reward': 5,
    'blind_trap_penalty': -2,
    'survival_bonus': 1,
    'leader_bonus': 10,
    'last_place_penalty': -10,
    'game_win_bonus': 50,
    'game_survival_bonus': 30
}

# Helper functions to get configurations
def get_agent_type(bot_id):
    """Get the agent type for a specific bot"""
    return BOT_CONFIGS.get(str(bot_id), AGENT_TYPES['MAIN_AGENT'])

def get_model_path(agent_type):
    """Get the model path for a specific agent type"""
    if agent_type == AGENT_TYPES['PLAYER_MODEL']:
        return MODEL_PATHS['player_model']
    elif agent_type == AGENT_TYPES['LSTM_DQN']:
        return MODEL_PATHS['lstm_dqn']
    elif agent_type == AGENT_TYPES['STANDARD_DQN']:
        return MODEL_PATHS['standard_dqn']
    else:
        return MODEL_PATHS['main_agent']

def get_training_config(agent_type):
    """Get training configuration for a specific agent type"""
    if agent_type == AGENT_TYPES['PLAYER_MODEL']:
        return TRAINING_CONFIG['player_model']
    elif agent_type == AGENT_TYPES['LSTM_DQN']:
        return TRAINING_CONFIG['lstm_dqn']
    elif agent_type == AGENT_TYPES['STANDARD_DQN']:
        return TRAINING_CONFIG['standard_dqn']
    else:
        # Return LSTM or standard config based on main agent type
        if USE_LSTM_DQN_FOR_MAIN:
            return TRAINING_CONFIG['lstm_dqn']
        else:
            return TRAINING_CONFIG['standard_dqn']

# Export useful lists for main.py
PLAYER_MODEL_BOTS = [int(bot_id) for bot_id, agent_type in BOT_CONFIGS.items() 
                     if agent_type == AGENT_TYPES['PLAYER_MODEL']]
LSTM_DQN_BOTS = [int(bot_id) for bot_id, agent_type in BOT_CONFIGS.items() 
                 if agent_type == AGENT_TYPES['LSTM_DQN']]
STANDARD_DQN_BOTS = [int(bot_id) for bot_id, agent_type in BOT_CONFIGS.items() 
                     if agent_type == AGENT_TYPES['STANDARD_DQN']]