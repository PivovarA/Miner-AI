# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

# Import from our modules
from config import *
from agents import DQNAgent
from game_state_processor import GameStateProcessor
from reward_calculator import RewardCalculator
from training_manager import TrainingManager
from utils import save_player_data, initialize_bot_state

# Import player data collector and behavior model
from player_data_collector import PlayerDataCollector
from player_behavior_model import PlayerBehaviorAgent

app = Flask(__name__)
CORS(app)

# Initialize main RL agent
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

# Initialize player behavior agent
player_agent = PlayerBehaviorAgent(STATE_SIZE, ACTION_SIZE)
player_agent.load_model('player_behavior_model.pth')

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
                
                # Store experience
                agent.memory.push(
                    bot_states[bot_id]['previous_state'], 
                    bot_states[bot_id]['previous_action'], 
                    reward, 
                    current_state, 
                    data.get('game_over', False)
                )
                
                # Train the agent
                agent.train()
            
            # Choose action based on bot configuration
            if USE_PLAYER_MODEL and bot_id in PLAYER_MODEL_BOTS:
                # Use player behavior model
                action_idx = player_agent.choose_action(current_state, use_epsilon=False)
            else:
                # Use regular RL agent
                action_idx = agent.choose_action(current_state)
            
            action = ACTIONS[action_idx]
            
            # Track actions for reward calculation
            bot_states[bot_id]['last_action'] = action
            
            # Store for next iteration
            bot_states[bot_id]['previous_state'] = current_state.copy()
            bot_states[bot_id]['previous_action'] = action_idx
        
        return jsonify({'action': action})
    
    except Exception as e:
        print(f"Error in get_action: {e}")
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
        
        # Train the player behavior model
        print(f"Training on {len(training_data['states'])} examples")
        player_agent.train_supervised(
            training_data['states'],
            training_data['actions'],
            epochs=10
        )
        
        # Save the trained model
        player_agent.save_model('player_behavior_model.pth')
        
        return jsonify({
            'status': 'success',
            'message': f'Trained on {len(training_data["states"])} examples'
        })
    
    except Exception as e:
        print(f"Error training player model: {e}")
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
                
                # Add ranking bonus
                final_scores = data.get('final_scores', [])
                if final_scores:
                    bot_rank = next((i for i, score in enumerate(final_scores) if score['id'] == bot_id), -1)
                    if bot_rank == 0:
                        final_reward += 50  # Big bonus for winning
                    elif bot_rank == 1:
                        final_reward += 20
                    elif bot_rank == 2:
                        final_reward += 5
                    else:
                        final_reward -= 20  # Penalty for last place
                
                # Survival bonus
                if data.get('bot_alive', False):
                    final_reward += 30
                else:
                    final_reward -= 30  # Penalty for dying
                
                # Store final experience
                current_state = GameStateProcessor.get_state_vector(data, STATE_SIZE)
                agent.memory.push(
                    bot_states[bot_id]['previous_state'], 
                    bot_states[bot_id]['previous_action'], 
                    final_reward, 
                    current_state, 
                    True
                )
                
                # Train the agent
                agent.train()
                
                # Update episode reward
                bot_states[bot_id]['episode_reward'] += final_reward
                
                print(f"Bot {bot_id} - Episode Reward: {bot_states[bot_id]['episode_reward']:.2f}")
            
            # Check if all bots have completed
            completed_bots = sum(1 for id in bot_states if bot_states[id].get('episode_reward', 0) != 0)
            if completed_bots >= len(bot_states) and completed_bots >= 4:  # All bots have completed
                agent.episodes += 1
                agent.update_epsilon()
                
                # Update episode stats
                avg_reward, total_rewards = training_manager.update_episode_stats(
                    bot_states, 
                    agent.epsilon
                )
                
                # Save statistics periodically
                if training_manager.current_episode % 5 == 0:
                    training_manager.save_training_stats(agent.epsilon)
                    agent.save_model()
                
                # Reset bot states
                for id in bot_states:
                    bot_states[id] = initialize_bot_state()
        
        return jsonify({'status': 'ok'})
    
    except Exception as e:
        print(f"Error in game_over: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("Starting RL Game Server with Player Support...")
    print("Agent initialized with:")
    print(f"  Episodes: {agent.episodes}")
    print(f"  Epsilon: {agent.epsilon}")
    print(f"  Learning Rate: {agent.learning_rate}")
    print(f"  Batch Size: {agent.batch_size}")
    print(f"  State Size: {STATE_SIZE}")
    print(f"  Player Model Enabled: {USE_PLAYER_MODEL}")
    if USE_PLAYER_MODEL:
        print(f"  Player Model Bots: {PLAYER_MODEL_BOTS}")
    
    # Run with threading disabled to avoid heap corruption
    app.run(host=HOST, port=PORT, threaded=False)