# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import numpy as np

# Import from our modules
from config import *
from agents import DQNAgent, LSTMDQNGameAgent
from game_state_processor import GameStateProcessor
from reward_calculator import RewardCalculator
from training_manager import TrainingManager
from utils import save_player_data, initialize_bot_state

# Import agent configuration
from agent_config import (
    USE_LSTM_DQN_FOR_MAIN, 
    PLAYER_MODEL_BOTS, 
    LSTM_DQN_BOTS, 
    STANDARD_DQN_BOTS,
    REWARD_CONFIG,
    get_agent_type,
    AGENT_TYPES
)

# Import player data collector and behavior model
from player_data_collector import PlayerDataCollector
from lstm_dqn_player_model import LSTMDQNAgent

app = Flask(__name__)
CORS(app)

# Initialize main RL agent based on configuration
if USE_LSTM_DQN_FOR_MAIN:
    main_agent = LSTMDQNGameAgent(STATE_SIZE, ACTION_SIZE)
else:
    main_agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

# Initialize additional agents for different bot types
lstm_agent = LSTMDQNGameAgent(STATE_SIZE, ACTION_SIZE)
standard_agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

# Initialize player behavior agent (LSTM-based)
player_agent = LSTMDQNAgent(STATE_SIZE, ACTION_SIZE)
player_agent.load_model('lstm_dqn_player_model.pth')

# Initialize player data collector
player_data_collector = PlayerDataCollector()

# Initialize training manager
training_manager = TrainingManager()

# Store previous states for all bots
bot_states = {}
lock = threading.Lock()

@app.route('/get_action', methods=['POST'])
def get_action():
    global bot_states
    
    try:
        data = request.json
        bot_id = data['bot_id']
        
        with lock:
            # Initialize bot tracking if needed
            if bot_id not in bot_states:
                bot_states[bot_id] = initialize_bot_state()
                # Reset episode for LSTM agents
                agent_type = get_agent_type(bot_id)
                if agent_type == AGENT_TYPES['LSTM_DQN']:
                    lstm_agent.reset_episode()
                elif agent_type == AGENT_TYPES['MAIN_AGENT'] and USE_LSTM_DQN_FOR_MAIN:
                    main_agent.reset_episode()
            
            # Get current state
            current_state = GameStateProcessor.get_state_vector(data, STATE_SIZE)
            
            # Compute reward if we have a previous state
            if bot_states[bot_id]['previous_state'] is not None:
                reward = RewardCalculator.calculate_reward(
                    data, 
                    bot_states[bot_id]['previous_state'], 
                    bot_id, 
                    bot_states
                )
                bot_states[bot_id]['episode_reward'] += reward
                
                # Store experience in appropriate agent
                agent_type = get_agent_type(bot_id)
                
                if agent_type == AGENT_TYPES['PLAYER_MODEL']:
                    # Player model doesn't train during gameplay
                    pass
                elif agent_type == AGENT_TYPES['LSTM_DQN']:
                    lstm_agent.memory.push(
                        bot_states[bot_id]['previous_state'], 
                        bot_states[bot_id]['previous_action'], 
                        reward, 
                        current_state, 
                        data.get('game_over', False)
                    )
                    lstm_agent.train()
                elif agent_type == AGENT_TYPES['STANDARD_DQN']:
                    standard_agent.memory.push(
                        bot_states[bot_id]['previous_state'], 
                        bot_states[bot_id]['previous_action'], 
                        reward, 
                        current_state, 
                        data.get('game_over', False)
                    )
                    standard_agent.train()
                else:  # MAIN_AGENT
                    main_agent.memory.push(
                        bot_states[bot_id]['previous_state'], 
                        bot_states[bot_id]['previous_action'], 
                        reward, 
                        current_state, 
                        data.get('game_over', False)
                    )
                    main_agent.train()
            
            # Choose action based on agent type
            agent_type = get_agent_type(bot_id)
            
            if agent_type == AGENT_TYPES['PLAYER_MODEL']:
                action_idx = player_agent.select_action(current_state, evaluation=True)
            elif agent_type == AGENT_TYPES['LSTM_DQN']:
                action_idx = lstm_agent.choose_action(current_state)
            elif agent_type == AGENT_TYPES['STANDARD_DQN']:
                action_idx = standard_agent.choose_action(current_state)
            else:  # MAIN_AGENT
                action_idx = main_agent.choose_action(current_state)
            
            action = ACTIONS[action_idx]
            
            # Track actions for reward calculation
            bot_states[bot_id]['last_action'] = action
            
            # Store for next iteration
            bot_states[bot_id]['previous_state'] = current_state.copy()
            bot_states[bot_id]['previous_action'] = action_idx
        
        return jsonify({'action': action})
    
    except Exception as e:
        print(f"Error in get_action: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'action': 'wait'})

@app.route('/save_player_data', methods=['POST'])
def save_player_data_route():
    try:
        data = request.json
        result = save_player_data(data)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error saving player data: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/train_player_model', methods=['POST'])
def train_player_model():
    try:
        # Load all player data
        collector = PlayerDataCollector()
        training_data = collector.get_training_data()
        
        if len(training_data['states']) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No player data available for training'
            })
        
        # Convert actions to indices for pretraining
        action_indices = []
        for action in training_data['actions']:
            if isinstance(action, str):
                action_indices.append(ACTIONS.index(action))
            else:
                action_indices.append(action)
        
        # Train the LSTM-DQN player model
        print(f"Training LSTM-DQN on {len(training_data['states'])} examples")
        player_agent.pretrain_on_player_data(
            states=training_data['states'],
            actions=np.array(action_indices),
            rewards=training_data['rewards'],
            next_states=training_data['next_states'],
            dones=training_data['dones'],
            epochs=10
        )
        
        # Save the trained model
        player_agent.save_model('lstm_dqn_player_model.pth')
        
        return jsonify({
            'status': 'success',
            'message': f'Trained LSTM-DQN on {len(training_data["states"])} examples'
        })
    
    except Exception as e:
        print(f"Error training player model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/game_over', methods=['POST'])
def game_over():
    global bot_states
    
    try:
        data = request.json
        bot_id = data['bot_id']
        
        with lock:
            if bot_id in bot_states and bot_states[bot_id]['previous_state'] is not None:
                # Calculate final reward
                final_reward = RewardCalculator.calculate_reward(
                    data, 
                    bot_states[bot_id]['previous_state'], 
                    bot_id, 
                    bot_states
                )
                
                # Add ranking bonus from configuration
                final_scores = data.get('final_scores', [])
                if final_scores:
                    bot_rank = next((i for i, score in enumerate(final_scores) if score['id'] == bot_id), -1)
                    if bot_rank == 0:
                        final_reward += REWARD_CONFIG['game_win_bonus']
                    elif bot_rank == 1:
                        final_reward += 20
                    elif bot_rank == 2:
                        final_reward += 5
                    elif bot_rank >= 3:
                        final_reward -= 20
                
                # Survival bonus
                if data.get('bot_alive', False):
                    final_reward += REWARD_CONFIG['game_survival_bonus']
                else:
                    final_reward -= REWARD_CONFIG['game_survival_bonus']
                
                # Store final experience
                current_state = GameStateProcessor.get_state_vector(data, STATE_SIZE)
                agent_type = get_agent_type(bot_id)
                
                if agent_type == AGENT_TYPES['PLAYER_MODEL']:
                    # Player model doesn't train during gameplay
                    pass
                elif agent_type == AGENT_TYPES['LSTM_DQN']:
                    lstm_agent.memory.push(
                        bot_states[bot_id]['previous_state'], 
                        bot_states[bot_id]['previous_action'], 
                        final_reward, 
                        current_state, 
                        True
                    )
                    lstm_agent.train()
                elif agent_type == AGENT_TYPES['STANDARD_DQN']:
                    standard_agent.memory.push(
                        bot_states[bot_id]['previous_state'], 
                        bot_states[bot_id]['previous_action'], 
                        final_reward, 
                        current_state, 
                        True
                    )
                    standard_agent.train()
                else:  # MAIN_AGENT
                    main_agent.memory.push(
                        bot_states[bot_id]['previous_state'], 
                        bot_states[bot_id]['previous_action'], 
                        final_reward, 
                        current_state, 
                        True
                    )
                    main_agent.train()
                
                # Update episode reward
                bot_states[bot_id]['episode_reward'] += final_reward
                
                print(f"Bot {bot_id} - Episode Reward: {bot_states[bot_id]['episode_reward']:.2f}")
            
            # Check if all bots have completed
            completed_bots = sum(1 for id in bot_states if 'episode_reward' in bot_states[id] and bot_states[id]['episode_reward'] != 0)
            if completed_bots >= min(len(bot_states), 4):  # All bots have completed
                # Update episodes for all agents
                main_agent.episodes += 1
                main_agent.update_epsilon()
                
                lstm_agent.episodes += 1
                lstm_agent.update_epsilon()
                
                standard_agent.episodes += 1
                standard_agent.update_epsilon()
                
                # Update episode stats
                avg_reward, total_rewards = training_manager.update_episode_stats(
                    bot_states, 
                    main_agent.epsilon
                )
                
                # Save statistics periodically
                if training_manager.current_episode % 5 == 0:
                    training_manager.save_training_stats(main_agent.epsilon)
                    main_agent.save_model()
                    lstm_agent.save_model('lstm_dqn_model.pth')
                    standard_agent.save_model('standard_dqn_model.pth')
                
                # Reset bot states for next episode
                for id in bot_states:
                    bot_states[id] = initialize_bot_state()
                    # Reset LSTM states for LSTM agents
                    agent_type = get_agent_type(id)
                    if agent_type == AGENT_TYPES['LSTM_DQN']:
                        lstm_agent.reset_episode()
                    elif agent_type == AGENT_TYPES['MAIN_AGENT'] and USE_LSTM_DQN_FOR_MAIN:
                        main_agent.reset_episode()
        
        return jsonify({'status': 'ok'})
    
    except Exception as e:
        print(f"Error in game_over: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("Starting RL Game Server with LSTM-DQN Support...")
    print("Agent Configuration:")
    print(f"  Main Agent Type: {'LSTM-DQN' if USE_LSTM_DQN_FOR_MAIN else 'Standard DQN'}")
    print(f"  Episodes: {main_agent.episodes}")
    print(f"  Epsilon: {main_agent.epsilon}")
    print(f"  Learning Rate: {main_agent.learning_rate}")
    print(f"  Batch Size: {main_agent.batch_size}")
    print(f"  State Size: {STATE_SIZE}")
    print("\nBot Configuration:")
    print(f"  Player Model Bots: {PLAYER_MODEL_BOTS}")
    print(f"  LSTM-DQN Bots: {LSTM_DQN_BOTS}")
    print(f"  Standard DQN Bots: {STANDARD_DQN_BOTS}")
    
    # Run with threading disabled to avoid heap corruption
    app.run(host=HOST, port=PORT, threaded=False)