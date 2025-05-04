# Treasure Hunter Game

A competitive multiplayer game where AI bots and human players compete to collect treasures while avoiding traps. The game features reinforcement learning AI that improves through gameplay and can learn from human player behavior.

![Game Screenshot](screenshot.png) <!-- Add an actual screenshot if available -->

## 🎮 Game Overview

In Treasure Hunter, players navigate a grid-based arena filled with obstacles, treasures, and traps. Each player has limited vision and must strategically:
- Collect treasures for points
- Deploy traps to eliminate opponents
- Avoid enemy traps
- Navigate obstacles
- Make decisions with limited information

### Key Features

- **Limited Vision**: Players can only see within a cone-shaped field of view
- **Strategic Combat**: Deploy and detonate traps to eliminate opponents
- **AI Learning**: Bots improve through reinforcement learning
- **Human Imitation**: AI can learn from human player behavior
- **Real-time Multiplayer**: Multiple bots compete simultaneously
- **Data Collection**: Records human gameplay for AI training

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js (for running the web server)
- Modern web browser

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/treasure-hunter-game.git
cd treasure-hunter-game
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir player_data
```

### Running the Game

1. Start the AI server:
```bash
python main.py
```

2. Open `index.html` in a web browser

3. Configure game settings:
   - Enable/disable human player
   - Set player name
   - Toggle data collection

## 🎯 How to Play

### Controls
- **Arrow Keys**: Move in four directions
- **Spacebar**: Drop a trap at current location
- **E**: Detonate all your placed traps
- **Restart Button**: Start a new game

### Game Rules
1. Collect green diamond treasures for points
2. Each player starts with 10 traps
3. Traps explode in a radius, eliminating players caught in the blast
4. The game ends when:
   - All treasures are collected
   - Time runs out (60 seconds)
   - Only one player remains alive
5. Score = (Treasures × 10) + (20 if alive)

### Vision System
- Players have limited cone-shaped vision
- Can only see treasures, opponents, and traps within vision range
- Must navigate strategically with incomplete information

## 🤖 AI System

### Reinforcement Learning
- Uses Deep Q-Network (DQN) for decision making
- Learns through experience replay
- Implements epsilon-greedy exploration

### State Representation
The AI perceives the game through a 69-feature vector including:
- Bot position and direction
- Visible treasures, enemies, and traps
- Game progress information
- Survival status

### Reward System
Positive rewards for:
- Collecting treasures (+50 per treasure)
- Spotting enemies (+5 per enemy)
- Eliminating opponents (+100 per kill)
- Strategic positioning

Penalties for:
- Dying (-100)
- Wasted explosions (-30)
- Being spotted (-5)
- Inactivity (-2)

### Human Behavior Imitation
- Collects data from human players
- Trains a separate neural network to mimic human strategies
- Can configure specific bots to use human-like behavior

## 📁 Project Structure

```
treasure-hunter-game/
├── index.html              # Game client
├── main.py                 # Flask server & API endpoints
├── config.py              # Game configuration
├── agents.py              # DQN implementation
├── models.py              # Neural network architectures
├── reward_calculator.py   # Reward computation logic
├── game_state_processor.py # State vector processing
├── training_manager.py    # Training statistics management
├── player_data_collector.py # Human data collection
├── player_behavior_model.py # Human imitation learning
├── train_player_model.py  # Training script for human behavior
└── utils.py               # Utility functions
```

## 🛠️ Configuration

### Game Settings (config.py)
```python
# Game constants
ACTIONS = ['move_up', 'move_down', 'move_left', 'move_right', 
           'drop_trap', 'explode_traps', 'wait']
STATE_SIZE = 69  # Neural network input size
ACTION_SIZE = 7  # Number of possible actions

# Bot configuration  
USE_PLAYER_MODEL = False  # Use human-trained models
PLAYER_MODEL_BOTS = [0, 1]  # Which bots use human behavior

# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor
```

### Client Settings (index.html)
```javascript
const NUM_BOTS = 4;          // Number of AI players
const GAME_DURATION = 60000; // Game length in ms
const DEVELOPER_MODE = false; // Show full map
const USE_SERVER_BOT = true; // Enable AI server
```

## 📊 Training the AI

### Training from Gameplay
The AI automatically learns during gameplay:
1. Bots play multiple games
2. Experience is stored in replay memory
3. Neural network updates after each action
4. Model saves every 5 episodes

### Training from Human Data
1. Enable human player and data collection
2. Play several games as a human
3. Run the training script:
```bash
python train_player_model.py --epochs 20 --data-dir player_data
```

### Monitoring Progress
- Training statistics are saved to `training_stats.json`
- Model checkpoints are saved to `dqn_model.pth`
- Human behavior model saved to `player_behavior_model.pth`

## 🔧 API Endpoints

### Game Server API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/get_action` | POST | Get AI action for current game state |
| `/game_over` | POST | Report game results for training |
| `/save_player_data` | POST | Save human player session data |
| `/train_player_model` | POST | Train model on collected human data |

### Request/Response Formats

#### Get Action Request
```json
{
  "bot_id": 0,
  "bot_position": {"x": 10, "y": 15},
  "bot_direction": 1.57,
  "visible_treasures": [...],
  "visible_bots": [...],
  "game_over": false
}
```

#### Get Action Response
```json
{
  "action": "move_up"
}
```

## 🎯 Advanced Features

### Custom Bot Behavior
Configure different bots with different strategies:
```python
# In config.py
PLAYER_MODEL_BOTS = [0, 1]  # Bots 0,1 use human strategy
# Bots 2,3 use RL strategy
```

### Data Analysis
Analyze collected player data:
```python
from player_data_collector import PlayerDataCollector

collector = PlayerDataCollector()
sessions = collector.load_all_sessions()

# Analyze player strategies
for session in sessions:
    print(f"Player: {session['player_id']}")
    print(f"Score: {session['final_score']}")
    print(f"Actions taken: {len(session['actions'])}")
```

### Custom Rewards
Modify `reward_calculator.py` to experiment with different reward schemes:
```python
# Emphasize aggressive play
if bot_states[bot_id]['last_action'] == 'explode_traps':
    if enemies_killed > 0:
        reward += 150  # Increased combat reward
```

## 🐛 Troubleshooting

### Common Issues

1. **Bots not moving**: Check if the Python server is running
2. **Game not starting**: Ensure all dependencies are installed
3. **Training not saving**: Check write permissions for model files
4. **Slow performance**: Reduce number of bots or disable developer mode

### Debug Mode
Enable detailed logging:
```python
# In main.py
app.debug = True
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Flask, PyTorch, and vanilla JavaScript
- Inspired by competitive multiplayer games
- Uses reinforcement learning techniques from DeepMind's DQN paper