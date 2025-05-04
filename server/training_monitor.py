import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time
import json
import os

class TrainingMonitor:
    def __init__(self):
        self.episode_stats = []
        self.load_stats()
        
    def load_stats(self):
        """Load stats from file"""
        if os.path.exists('training_stats.json'):
            try:
                with open('training_stats.json', 'r') as f:
                    data = json.load(f)
                    self.episode_stats = data.get('episode_stats', [])
            except json.JSONDecodeError:
                print("Warning: Could not parse training_stats.json")
                self.episode_stats = []
        else:
            self.episode_stats = []
    
    def plot(self):
        if not self.episode_stats:
            print("No data to plot yet...")
            return
            
        episodes = [stat['episode'] for stat in self.episode_stats]
        avg_rewards = [stat['avg_reward'] for stat in self.episode_stats]
        total_rewards = [stat['total_reward'] for stat in self.episode_stats]
        epsilons = [stat['epsilon'] for stat in self.episode_stats]
        
        plt.figure(figsize=(15, 10))
        
        # Plot average rewards
        plt.subplot(2, 2, 1)
        plt.plot(episodes, avg_rewards, 'b-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Average Reward per Episode')
        plt.grid(True)
        
        # Plot total rewards
        plt.subplot(2, 2, 2)
        plt.plot(episodes, total_rewards, 'g-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.grid(True)
        
        # Plot epsilon decay
        plt.subplot(2, 2, 3)
        plt.plot(episodes, epsilons, 'r-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate (Epsilon)')
        plt.grid(True)
        
        # Plot individual bot performance
        plt.subplot(2, 2, 4)
        if len(self.episode_stats) > 0:
            recent_stats = self.episode_stats[-50:]  # Last 50 episodes
            bot_rewards = {}
            
            for stat in recent_stats:
                for bot_id, reward in stat['individual_rewards'].items():
                    if bot_id not in bot_rewards:
                        bot_rewards[bot_id] = []
                    bot_rewards[bot_id].append(reward)
            
            for bot_id, rewards in bot_rewards.items():
                plt.plot(range(len(rewards)), rewards, label=f'Bot {bot_id}', linewidth=2)
            
            plt.xlabel('Recent Episodes')
            plt.ylabel('Individual Rewards')
            plt.title('Individual Bot Performance (Last 50 Episodes)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150)
        plt.close()
        
        # Create additional visualizations
        self.plot_statistics()
    
    def plot_statistics(self):
        if not self.episode_stats:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Calculate win rates for each bot position
        win_counts = {}
        
        for stat in self.episode_stats:
            # Determine winner based on highest reward
            rewards = stat['individual_rewards']
            if rewards:
                winner = max(rewards.items(), key=lambda x: x[1])[0]
                win_counts[winner] = win_counts.get(winner, 0) + 1
        
        if win_counts:
            # Plot win distribution
            plt.subplot(2, 1, 1)
            bot_ids = list(win_counts.keys())
            wins = [win_counts[bid] for bid in bot_ids]
            plt.bar(bot_ids, wins)
            plt.xlabel('Bot ID')
            plt.ylabel('Number of Wins')
            plt.title('Win Distribution by Bot ID')
        
        # Plot learning curve (moving average)
        plt.subplot(2, 1, 2)
        window_size = 10
        if len(self.episode_stats) >= window_size:
            avg_rewards = [stat['avg_reward'] for stat in self.episode_stats]
            moving_avg = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(avg_rewards)), moving_avg, 'b-', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Moving Average Reward')
            plt.title(f'Learning Curve ({window_size}-Episode Moving Average)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_statistics.png', dpi=150)
        plt.close()

    def print_summary(self):
        if not self.episode_stats:
            print("No training data available")
            return
            
        total_episodes = len(self.episode_stats)
        recent_stats = self.episode_stats[-10:] if total_episodes >= 10 else self.episode_stats
        
        if recent_stats:
            avg_recent_reward = np.mean([stat['avg_reward'] for stat in recent_stats])
            total_recent_reward = np.mean([stat['total_reward'] for stat in recent_stats])
            current_epsilon = self.episode_stats[-1]['epsilon'] if self.episode_stats else 1.0
            
            print(f"\n=== Training Summary ===")
            print(f"Total Episodes: {total_episodes}")
            print(f"Current Epsilon: {current_epsilon:.3f}")
            print(f"Recent Avg Reward (last 10): {avg_recent_reward:.2f}")
            print(f"Recent Total Reward (last 10): {total_recent_reward:.2f}")
            
            if self.episode_stats:
                last_episode = self.episode_stats[-1]
                print(f"\nLast Episode Results:")
                for bot_id, reward in last_episode['individual_rewards'].items():
                    print(f"  Bot {bot_id}: {reward:.2f}")
        else:
            print("Not enough data to display summary")

if __name__ == "__main__":
    monitor = TrainingMonitor()
    
    print("Training Monitor started. Reading from training_stats.json...")
    print("Press Ctrl+C to stop.")
    
    while True:
        try:
            # Reload stats from file
            monitor.load_stats()
            
            # Generate plots
            monitor.plot()
            
            # Print summary
            monitor.print_summary()
            
            # Wait before next update
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\nStopping monitor...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)