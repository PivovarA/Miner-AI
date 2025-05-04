# lstm_dqn_training_manager.py
import json
import os
import numpy as np
from datetime import datetime

class LSTMDQNTrainingManager:
    def __init__(self, save_dir='training_results'):
        self.save_dir = save_dir
        self.episode_stats = []
        self.current_episode = 0
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'epsilon_values': [],
            'q_values': []
        }
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.load_training_stats()
    
    def save_training_stats(self, agent_epsilon=None):
        """Save training statistics"""
        try:
            stats = {
                'episodes': self.current_episode,
                'episode_stats': self.episode_stats[-1000:],  # Keep last 1000 episodes
                'training_history': self.training_history,
                'epsilon': agent_epsilon,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            stats_path = os.path.join(self.save_dir, 'lstm_dqn_training_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Training stats saved to {stats_path}")
        except Exception as e:
            print(f"Error saving training stats: {e}")
    
    def load_training_stats(self):
        """Load training statistics"""
        stats_path = os.path.join(self.save_dir, 'lstm_dqn_training_stats.json')
        
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    self.current_episode = stats.get('episodes', 0)
                    self.episode_stats = stats.get('episode_stats', [])
                    self.training_history = stats.get('training_history', {
                        'episodes': [], 'rewards': [], 'losses': [], 
                        'epsilon_values': [], 'q_values': []
                    })
                    print(f"Loaded training stats. Episodes: {self.current_episode}")
            except Exception as e:
                print(f"Error loading training stats: {e}")
                self._initialize_empty_stats()
        else:
            self._initialize_empty_stats()
    
    def _initialize_empty_stats(self):
        """Initialize empty statistics"""
        self.current_episode = 0
        self.episode_stats = []
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'epsilon_values': [],
            'q_values': []
        }
    
    def update_episode_stats(self, episode_reward, episode_loss, agent_epsilon, 
                           q_values=None, actions_taken=None):
        """Update statistics for an episode"""
        self.current_episode += 1
        
        # Calculate average loss for the episode
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        
        # Calculate average Q-value if provided
        avg_q_value = np.mean(q_values) if q_values is not None else 0
        
        # Create episode stats
        episode_stat = {
            'episode': self.current_episode,
            'reward': episode_reward,
            'avg_loss': avg_loss,
            'epsilon': agent_epsilon,
            'avg_q_value': avg_q_value,
            'actions_taken': actions_taken if actions_taken else []
        }
        
        # Add to history
        self.episode_stats.append(episode_stat)
        self.training_history['episodes'].append(self.current_episode)
        self.training_history['rewards'].append(episode_reward)
        self.training_history['losses'].append(avg_loss)
        self.training_history['epsilon_values'].append(agent_epsilon)
        self.training_history['q_values'].append(avg_q_value)
        
        # Log progress
        print(f"\nEpisode {self.current_episode} complete:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Epsilon: {agent_epsilon:.3f}")
        print(f"  Avg Q-value: {avg_q_value:.4f}")
        
        # Save stats periodically
        if self.current_episode % 10 == 0:
            self.save_training_stats(agent_epsilon)
    
    def get_training_summary(self):
        """Get a summary of training progress"""
        if not self.episode_stats:
            return "No training data available."
        
        recent_stats = self.episode_stats[-100:]  # Last 100 episodes
        
        summary = {
            'total_episodes': self.current_episode,
            'avg_reward_last_100': np.mean([s['reward'] for s in recent_stats]),
            'avg_loss_last_100': np.mean([s['avg_loss'] for s in recent_stats]),
            'current_epsilon': recent_stats[-1]['epsilon'] if recent_stats else 0,
            'avg_q_value_last_100': np.mean([s['avg_q_value'] for s in recent_stats])
        }
        
        return summary
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        import matplotlib.pyplot as plt
        
        if not self.training_history['episodes']:
            print("No training data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        ax1.plot(self.training_history['episodes'], self.training_history['rewards'])
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.grid(True)
        
        # Plot losses
        ax2.plot(self.training_history['episodes'], self.training_history['losses'])
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Average Episode Loss')
        ax2.grid(True)
        
        # Plot epsilon
        ax3.plot(self.training_history['episodes'], self.training_history['epsilon_values'])
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('Exploration Rate (Epsilon)')
        ax3.grid(True)
        
        # Plot Q-values
        ax4.plot(self.training_history['episodes'], self.training_history['q_values'])
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Q-value')
        ax4.set_title('Average Q-values')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'lstm_dqn_training_curves.png')
        
        plt.savefig(save_path)
        plt.close()
        print(f"Training curves saved to {save_path}")
    
    def export_metrics(self, export_path=None):
        """Export training metrics to CSV"""
        import pandas as pd
        
        if not self.episode_stats:
            print("No data to export")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.episode_stats)
        
        if export_path is None:
            export_path = os.path.join(self.save_dir, 'lstm_dqn_training_metrics.csv')
        
        df.to_csv(export_path, index=False)
        print(f"Metrics exported to {export_path}")
    
    def analyze_action_distribution(self):
        """Analyze the distribution of actions taken"""
        if not self.episode_stats:
            return None
        
        action_counts = {}
        total_actions = 0
        
        for episode in self.episode_stats:
            actions = episode.get('actions_taken', [])
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1
                total_actions += 1
        
        if total_actions == 0:
            return None
        
        action_distribution = {}
        for action, count in action_counts.items():
            action_distribution[action] = {
                'count': count,
                'percentage': (count / total_actions) * 100
            }
        
        return action_distribution
    
    def get_performance_trend(self, window_size=10):
        """Get performance trend over time"""
        if len(self.episode_stats) < window_size:
            return None
        
        rewards = [stat['reward'] for stat in self.episode_stats]
        
        # Calculate moving average
        moving_avg = []
        for i in range(window_size, len(rewards) + 1):
            window = rewards[i-window_size:i]
            moving_avg.append(sum(window) / window_size)
        
        # Calculate trend (positive = improving, negative = declining)
        if len(moving_avg) >= 2:
            trend = (moving_avg[-1] - moving_avg[0]) / len(moving_avg)
        else:
            trend = 0
        
        return {
            'moving_average': moving_avg,
            'trend': trend,
            'latest_average': moving_avg[-1] if moving_avg else 0
        }
    
    def create_training_report(self):
        """Create a comprehensive training report"""
        summary = self.get_training_summary()
        action_dist = self.analyze_action_distribution()
        performance_trend = self.get_performance_trend()
        
        report = f"""
LSTM-DQN Training Report
=======================

Training Summary:
----------------
Total Episodes: {summary['total_episodes']}
Average Reward (Last 100): {summary['avg_reward_last_100']:.2f}
Average Loss (Last 100): {summary['avg_loss_last_100']:.4f}
Current Epsilon: {summary['current_epsilon']:.3f}
Average Q-value (Last 100): {summary['avg_q_value_last_100']:.4f}

Performance Trend:
-----------------
"""
        
        if performance_trend:
            report += f"Latest Moving Average: {performance_trend['latest_average']:.2f}\n"
            report += f"Trend: {'Improving' if performance_trend['trend'] > 0 else 'Declining'} ({performance_trend['trend']:.4f})\n"
        else:
            report += "Insufficient data for trend analysis\n"
        
        report += "\nAction Distribution:\n------------------\n"
        
        if action_dist:
            for action, stats in action_dist.items():
                report += f"{action}: {stats['count']} times ({stats['percentage']:.1f}%)\n"
        else:
            report += "No action data available\n"
        
        return report

# Example usage functions for integration with the game

def integrate_with_game(agent, training_manager):
    """Example function showing how to integrate with a game loop"""
    
    # This would be called at the start of an episode
    agent.reset_episode()
    episode_reward = 0
    episode_losses = []
    episode_q_values = []
    episode_actions = []
    
    # During the game loop, for each step:
    def game_step(current_state):
        nonlocal episode_reward, episode_losses, episode_q_values, episode_actions
        
        # Select action
        action = agent.select_action(current_state)
        episode_actions.append(action)
        
        # Execute action in game and get results
        # next_state, reward, done = game.step(action)  # This would be your game logic
        
        # For demonstration, creating dummy values
        next_state = current_state  # In reality, this would come from the game
        reward = 0.1  # In reality, this would come from the game
        done = False  # In reality, this would come from the game
        
        # Store transition
        agent.store_transition(current_state, action, next_state, reward, done)
        
        # Train the agent
        if len(agent.memory) > 64:  # batch_size
            loss = agent.train(64)
            if loss is not None:
                episode_losses.append(loss)
        
        episode_reward += reward
        
        # Optionally collect Q-values for analysis
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(agent.device)
            q_values, _ = agent.policy_net(state_tensor)
            episode_q_values.append(q_values.max().item())
        
        return next_state, done
    
    # At the end of an episode:
    def end_episode():
        # Update training statistics
        training_manager.update_episode_stats(
            episode_reward=episode_reward,
            episode_loss=episode_losses,
            agent_epsilon=agent.epsilon,
            q_values=episode_q_values,
            actions_taken=episode_actions
        )
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Periodically update target network
        if training_manager.current_episode % 10 == 0:
            agent.update_target_network()
        
        # Save model periodically
        if training_manager.current_episode % 50 == 0:
            agent.save_model(f'checkpoints/lstm_dqn_episode_{training_manager.current_episode}.pth')
    
    return game_step, end_episode

# Main training script that demonstrates the full integration
def train_with_game_integration():
    """Complete training script with game integration"""
    import torch
    from lstm_dqn_player_model import LSTMDQNAgent, ACTIONS
    
    # Initialize agent and training manager
    state_size = 69  # Based on your game's state representation
    action_size = len(ACTIONS)
    
    agent = LSTMDQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        hidden_size=128,
        lstm_layers=2,
        sequence_length=10
    )
    
    training_manager = LSTMDQNTrainingManager(save_dir='training_results')
    
    # Create integration functions
    game_step, end_episode = integrate_with_game(agent, training_manager)
    
    # Training loop
    num_episodes = 1000
    max_steps_per_episode = 200
    
    for episode in range(num_episodes):
        # Reset the game and get initial state
        # In reality, this would be: initial_state = game.reset()
        initial_state = np.random.rand(state_size)  # Dummy initial state
        
        current_state = initial_state
        
        for step in range(max_steps_per_episode):
            next_state, done = game_step(current_state)
            current_state = next_state
            
            if done:
                break
        
        end_episode()
        
        # Generate reports periodically
        if episode % 100 == 0:
            training_manager.plot_training_curves()
            report = training_manager.create_training_report()
            print(report)
    
    # Final report
    training_manager.export_metrics()
    final_report = training_manager.create_training_report()
    
    with open('training_results/final_report.txt', 'w') as f:
        f.write(final_report)
    
    return agent, training_manager