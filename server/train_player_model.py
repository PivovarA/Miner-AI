#!/usr/bin/env python3
"""
Train Player Model Script

This script trains a model on collected player data to imitate human behavior.
Can be run standalone or integrated into the game server.
"""

import argparse
import os
import matplotlib.pyplot as plt
from player_data_collector import PlayerDataCollector
from player_behavior_model import PlayerBehaviorAgent, ACTIONS
import numpy as np

def visualize_training_data(states, actions):
    """Visualize the distribution of actions in the training data"""
    action_counts = {}
    for action in actions:
        if isinstance(action, str):
            action_name = action
        else:
            action_name = ACTIONS[action]
        
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    
    plt.bar(actions, counts)
    plt.xlabel('Actions')
    plt.ylabel('Count')
    plt.title('Distribution of Player Actions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('player_action_distribution.png')
    plt.close()
    
    print("\nAction Distribution:")
    for action, count in action_counts.items():
        print(f"  {action}: {count} ({count/len(states)*100:.1f}%)")

def train_player_model(data_dir='player_data', model_path='player_behavior_model.pth', 
                      epochs=20, batch_size=64, learning_rate=0.001):
    """Train the player behavior model on collected data"""
    
    print(f"Loading player data from {data_dir}...")
    collector = PlayerDataCollector(data_dir)
    training_data = collector.get_training_data()
    
    if len(training_data['states']) == 0:
        print("No player data found. Please collect some gameplay data first.")
        return
    
    print(f"Found {len(training_data['states'])} training examples")
    
    # Visualize the data
    visualize_training_data(training_data['states'], training_data['actions'])
    
    # Initialize the player behavior agent
    state_size = 69  # Updated to match the game's state size
    action_size = len(ACTIONS)
    
    player_agent = PlayerBehaviorAgent(state_size, action_size, learning_rate=learning_rate)
    
    # Load existing model if available
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        player_agent.load_model(model_path)
    
    # Train the model
    print(f"\nTraining player behavior model for {epochs} epochs...")
    player_agent.train_supervised(
        training_data['states'],
        training_data['actions'],
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Save the trained model
    player_agent.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Test the model accuracy
    test_accuracy(player_agent, training_data['states'], training_data['actions'])
    
    return player_agent

def test_accuracy(agent, states, actions):
    """Test the accuracy of the trained model"""
    correct = 0
    total = len(states)
    
    for state, action in zip(states, actions):
        predicted_action_idx = agent.choose_action(state, use_epsilon=False)
        
        if isinstance(action, str):
            actual_action_idx = ACTIONS.index(action)
        else:
            actual_action_idx = action
        
        if predicted_action_idx == actual_action_idx:
            correct += 1
    
    accuracy = correct / total
    print(f"\nModel Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Train player behavior model from collected gameplay data')
    parser.add_argument('--data-dir', default='player_data', help='Directory containing player data')
    parser.add_argument('--model-path', default='player_behavior_model.pth', help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    train_player_model(
        data_dir=args.data_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == '__main__':
    main()