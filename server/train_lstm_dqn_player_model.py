#!/usr/bin/env python3
"""
Train LSTM-DQN Player Model Script

This script trains an LSTM-DQN model on collected player data to imitate and improve upon human behavior.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from player_data_collector import PlayerDataCollector
from lstm_dqn_player_model import LSTMDQNAgent, ACTIONS

def visualize_training_progress(episode_rewards, episode_losses):
    """Visualize training progress"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.plot(episode_rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress - Episode Rewards')
    ax1.grid(True)
    
    # Plot losses
    if episode_losses:
        ax2.plot(episode_losses)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('lstm_dqn_training_progress.png')
    plt.close()

def convert_actions_to_indices(actions):
    """Convert string actions to indices"""
    indices = []
    for action in actions:
        if isinstance(action, str):
            indices.append(ACTIONS.index(action))
        else:
            indices.append(action)
    return np.array(indices)

def train_lstm_dqn_model(data_dir='player_data', model_path='lstm_dqn_player_model.pth',
                        pretrain_epochs=10, train_episodes=100, batch_size=64,
                        learning_rate=0.001, target_update_frequency=10):
    """Train the LSTM-DQN player behavior model"""
    
    print(f"Loading player data from {data_dir}...")
    collector = PlayerDataCollector(data_dir)
    training_data = collector.get_training_data()
    
    if len(training_data['states']) == 0:
        print("No player data found. Please collect some gameplay data first.")
        return
    
    print(f"Found {len(training_data['states'])} training examples")
    
    # Initialize the LSTM-DQN agent
    state_size = 69  # Matching the game's state size
    action_size = len(ACTIONS)
    
    agent = LSTMDQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        hidden_size=128,
        lstm_layers=2,
        sequence_length=10
    )
    
    # Load existing model if available
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        agent.load_model(model_path)
    
    # Convert actions to indices
    action_indices = convert_actions_to_indices(training_data['actions'])
    
    # Pretrain on player data
    print("\nPhase 1: Pretraining on player data...")
    agent.pretrain_on_player_data(
        states=training_data['states'],
        actions=action_indices,
        rewards=training_data['rewards'],
        next_states=training_data['next_states'],
        dones=training_data['dones'],
        epochs=pretrain_epochs
    )
    
    # Save after pretraining
    pretrain_path = model_path.replace('.pth', '_pretrained.pth')
    agent.save_model(pretrain_path)
    print(f"Pretrained model saved to {pretrain_path}")
    
    # Fine-tune with reinforcement learning
    print("\nPhase 2: Fine-tuning with reinforcement learning...")
    episode_rewards = []
    episode_losses = []
    
    for episode in range(train_episodes):
        episode_reward = 0
        episode_loss = []
        
        # Reset agent for new episode
        agent.reset_episode()
        
        # Use a random starting point from the data
        start_idx = np.random.randint(0, len(training_data['states']) - 100)
        
        # Run episode
        for i in range(start_idx, min(start_idx + 100, len(training_data['states']) - 1)):
            state = training_data['states'][i]
            next_state = training_data['next_states'][i]
            reward = training_data['rewards'][i]
            done = training_data['dones'][i]
            
            # Select action
            action = agent.select_action(state)
            
            # Store transition
            agent.store_transition(state, action, next_state, reward, done)
            
            # Train
            if len(agent.memory) > batch_size:
                loss = agent.train(batch_size)
                if loss is not None:
                    episode_loss.append(loss)
            
            episode_reward += reward
            
            if done:
                break
        
        # Update target network periodically
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record statistics
        episode_rewards.append(episode_reward)
        if episode_loss:
            episode_losses.extend(episode_loss)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            avg_loss = np.mean(episode_losses[-100:]) if episode_losses else 0
            print(f"Episode {episode}/{train_episodes} - "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Save the final model
    agent.save_model(model_path)
    print(f"\nFinal model saved to {model_path}")
    
    # Visualize training progress
    visualize_training_progress(episode_rewards, episode_losses)
    
    # Test the model
    test_model_performance(agent, training_data)
    
    return agent

def test_model_performance(agent, test_data):
    """Test the trained model's performance"""
    print("\nTesting model performance...")
    
    total_reward = 0
    action_distribution = {action: 0 for action in ACTIONS}
    
    # Test on a subset of data
    test_size = min(1000, len(test_data['states']))
    
    for i in range(test_size):
        state = test_data['states'][i]
        actual_action = test_data['actions'][i]
        reward = test_data['rewards'][i]
        
        # Get model's action
        predicted_action = agent.select_action(state, evaluation=True)
        action_name = ACTIONS[predicted_action]
        action_distribution[action_name] += 1
        
        total_reward += reward
    
    print(f"Average reward on test data: {total_reward/test_size:.4f}")
    print("\nAction distribution:")
    for action, count in action_distribution.items():
        print(f"  {action}: {count} ({count/test_size*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Train LSTM-DQN player behavior model')
    parser.add_argument('--data-dir', default='player_data', help='Directory containing player data')
    parser.add_argument('--model-path', default='lstm_dqn_player_model.pth', help='Path to save the trained model')
    parser.add_argument('--pretrain-epochs', type=int, default=10, help='Number of pretraining epochs')
    parser.add_argument('--train-episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--target-update-frequency', type=int, default=10, help='Target network update frequency')
    
    args = parser.parse_args()
    
    train_lstm_dqn_model(
        data_dir=args.data_dir,
        model_path=args.model_path,
        pretrain_epochs=args.pretrain_epochs,
        train_episodes=args.train_episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        target_update_frequency=args.target_update_frequency
    )

if __name__ == '__main__':
    main()