# player_data_collector.py
import json
import os
import time
from datetime import datetime
import numpy as np

class PlayerDataCollector:
    def __init__(self, data_dir="player_data"):
        self.data_dir = data_dir
        self.current_session = []
        self.session_id = None
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def start_session(self, player_id="human"):
        """Start a new data collection session"""
        self.session_id = f"{player_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = []
        self.player_id = player_id
        return self.session_id
    
    def record_action(self, state, action, reward=0, additional_info=None):
        """Record a single action with its context"""
        if self.session_id is None:
            self.start_session()
        
        record = {
            'timestamp': time.time(),
            'state': state,
            'action': action,
            'reward': reward,
            'additional_info': additional_info or {}
        }
        
        self.current_session.append(record)
    
    def end_session(self, final_score=None, won=False):
        """End the current session and save data"""
        if not self.current_session:
            return
        
        session_data = {
            'session_id': self.session_id,
            'player_id': self.player_id,
            'start_time': self.current_session[0]['timestamp'],
            'end_time': self.current_session[-1]['timestamp'],
            'duration': self.current_session[-1]['timestamp'] - self.current_session[0]['timestamp'],
            'total_actions': len(self.current_session),
            'final_score': final_score,
            'won': won,
            'actions': self.current_session
        }
        
        # Save session data
        filename = os.path.join(self.data_dir, f"{self.session_id}.json")
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Reset for next session
        self.current_session = []
        self.session_id = None
        
        return filename
    
    def load_all_sessions(self):
        """Load all recorded sessions"""
        sessions = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_dir, filename), 'r') as f:
                    sessions.append(json.load(f))
        return sessions
    
    def get_training_data(self):
        """Prepare data for training - handles both old and new data formats"""
        sessions = self.load_all_sessions()
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for session in sessions:
            # Handle both format types (with or without 'reward' field)
            session_actions = session.get('actions', [])
            
            for i in range(len(session_actions) - 1):
                current = session_actions[i]
                next_action = session_actions[i + 1]
                
                # Extract state data
                if 'state' in current:
                    state = current['state']
                else:
                    # For data from the frontend
                    state = self._convert_frontend_state_to_vector(current)
                
                states.append(state)
                actions.append(current['action'])
                
                # Handle reward field gracefully
                rewards.append(current.get('reward', 0))
                
                # Extract next state
                if 'state' in next_action:
                    next_state = next_action['state']
                else:
                    next_state = self._convert_frontend_state_to_vector(next_action)
                
                next_states.append(next_state)
                dones.append(False)
            
            # Handle the last action
            if session_actions:
                last_action = session_actions[-1]
                
                if 'state' in last_action:
                    state = last_action['state']
                    next_state = last_action['state']
                else:
                    state = self._convert_frontend_state_to_vector(last_action)
                    next_state = state  # Use same state for terminal state
                
                states.append(state)
                actions.append(last_action['action'])
                rewards.append(last_action.get('reward', 0))
                next_states.append(next_state)
                dones.append(True)
        
        # Convert states to numpy arrays, handling different formats
        processed_states = []
        processed_next_states = []
        
        for state in states:
            if isinstance(state, dict):
                processed_states.append(self._state_dict_to_vector(state))
            else:
                processed_states.append(state)
        
        for state in next_states:
            if isinstance(state, dict):
                processed_next_states.append(self._state_dict_to_vector(state))
            else:
                processed_next_states.append(state)
        
        return {
            'states': np.array(processed_states),
            'actions': actions,  # Keep as list of strings for flexibility
            'rewards': np.array(rewards),
            'next_states': np.array(processed_next_states),
            'dones': np.array(dones)
        }
    
    def _convert_frontend_state_to_vector(self, action_data):
        """Convert frontend action data to state dict format"""
        if 'state' in action_data:
            return action_data['state']
        
        # Create state dict from frontend data
        state = {
            'bot_position': action_data.get('position', {'x': 0, 'y': 0}),
            'bot_direction': action_data.get('direction', 0),
            'bot_treasures': action_data.get('treasures', 0),
            'bot_remaining_traps': action_data.get('remainingTraps', 0),
            'bot_alive': True  # Assume alive if recording actions
        }
        
        return state
    
    def _state_dict_to_vector(self, state_dict):
        """Convert state dictionary to feature vector matching game's state representation"""
        features = []
        
        # Bot position (normalized)
        features.extend([
            state_dict.get('bot_position', {}).get('x', 0) / 40,
            state_dict.get('bot_position', {}).get('y', 0) / 40
        ])
        
        # Bot direction (sin and cos)
        direction = state_dict.get('bot_direction', 0)
        features.extend([
            np.sin(direction),
            np.cos(direction)
        ])
        
        # Bot stats
        features.extend([
            state_dict.get('bot_treasures', 0) / 5,  # Assuming max 5 treasures
            state_dict.get('bot_remaining_traps', 0) / 10,
            float(state_dict.get('bot_alive', True))
        ])
        
        # Game progress
        features.extend([
            state_dict.get('treasures_collected', 0) / 5,
            state_dict.get('time_remaining', 60000) / 60000
        ])
        
        # Visible treasures (up to 5)
        visible_treasures = state_dict.get('visible_treasures', [])[:5]
        for i in range(5):
            if i < len(visible_treasures):
                features.extend([
                    visible_treasures[i].get('x', 0) / 10,
                    visible_treasures[i].get('y', 0) / 10
                ])
            else:
                features.extend([0, 0])
        
        # Visible bots (up to 3)
        visible_bots = state_dict.get('visible_bots', [])[:3]
        for i in range(3):
            if i < len(visible_bots):
                features.extend([
                    visible_bots[i].get('x', 0) / 10,
                    visible_bots[i].get('y', 0) / 10,
                    visible_bots[i].get('treasures', 0) / 5,
                    visible_bots[i].get('traps_remaining', 0) / 10
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        # Visible traps (up to 3)
        visible_traps = state_dict.get('visible_traps', [])[:3]
        for i in range(3):
            if i < len(visible_traps):
                features.extend([
                    visible_traps[i].get('x', 0) / 10,
                    visible_traps[i].get('y', 0) / 10,
                    1 if visible_traps[i].get('owner') == 'self' else 0
                ])
            else:
                features.extend([0, 0, 0])
        
        # Obstacle grid (5x5)
        visible_obstacles = state_dict.get('visible_obstacles', [])
        obstacle_grid = np.zeros(25)
        for obstacle in visible_obstacles:
            x, y = obstacle.get('x', 0) + 2, obstacle.get('y', 0) + 2
            if 0 <= x < 5 and 0 <= y < 5:
                obstacle_grid[y * 5 + x] = 1
        features.extend(obstacle_grid.tolist())
        
        # Bot scores
        bot_scores = state_dict.get('bot_scores', [])
        score_features = []
        for score in bot_scores[:3]:  # Take top 3 scores
            if score.get('id') != 'player':
                score_features.append(score.get('treasures', 0) / 5)
        while len(score_features) < 3:
            score_features.append(0)
        features.extend(score_features)
        
        # Explosion available
        features.append(float(state_dict.get('explosions_available', False)))
        
        # Ensure we have exactly 69 features
        while len(features) < 69:
            features.append(0)
        
        return features[:69]