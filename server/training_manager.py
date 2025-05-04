# training_manager.py
import json
import os

class TrainingManager:
    def __init__(self):
        self.episode_stats = []
        self.current_episode = 0
        self.load_training_stats()
    
    def save_training_stats(self, agent_epsilon):
        try:
            stats = {
                'episodes': self.current_episode,
                'episode_stats': self.episode_stats[-100:],  # Keep only last 100 episodes
                'epsilon': agent_epsilon
            }
            with open('training_stats.json', 'w') as f:
                json.dump(stats, f)
        except Exception as e:
            print(f"Error saving training stats: {e}")
    
    def load_training_stats(self):
        if os.path.exists('training_stats.json'):
            try:
                with open('training_stats.json', 'r') as f:
                    stats = json.load(f)
                    self.current_episode = stats.get('episodes', 0)
                    self.episode_stats = stats.get('episode_stats', [])
                    print(f"Loaded training stats. Episodes: {self.current_episode}")
            except Exception as e:
                print(f"Error loading training stats: {e}")
                self.current_episode = 0
                self.episode_stats = []
    
    def update_episode_stats(self, bot_states, agent_epsilon):
        self.current_episode += 1
        
        # Calculate episode statistics
        total_rewards = sum(bot_states[id]['reward'] for id in bot_states)
        avg_reward = total_rewards / max(1, len(bot_states))
        
        self.episode_stats.append({
            'episode': self.current_episode,
            'total_reward': total_rewards,
            'avg_reward': avg_reward,
            'epsilon': agent_epsilon,
            'individual_rewards': {id: bot_states[id]['reward'] for id in bot_states}
        })
        
        # Log progress
        print(f"\nEpisode {self.current_episode} complete:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Total Reward: {total_rewards:.2f}")
        print(f"  Epsilon: {agent_epsilon:.3f}")
        
        return avg_reward, total_rewards