# utils.py
import os
import json
from datetime import datetime

def save_player_data(data):
    try:
        player_name = data.get('player_name', 'unknown')
        
        # Create a unique session ID
        session_id = f"{player_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save the data
        filename = os.path.join('player_data', f"{session_id}.json")
        os.makedirs('player_data', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved player data to {filename}")
        
        return {
            'status': 'success',
            'session_id': session_id,
            'filename': filename
        }
    
    except Exception as e:
        print(f"Error saving player data: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

def initialize_bot_state():
    return {
        'previous_state': None,
        'previous_action': None,
        'episode_reward': 0,
        'last_action': 'wait',
        'steps_without_treasure': 0
    }