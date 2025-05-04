# lstm_dqn_player_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os

# Define actions to match the game
ACTIONS = ['move_up', 'move_down', 'move_left', 'move_right', 'drop_trap', 'explode_traps', 'wait']

# Define experience replay
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class LSTMDQNetwork(nn.Module):
    """LSTM-based Deep Q-Network for player behavior"""
    def __init__(self, state_size, action_size, hidden_size=128, lstm_layers=2):
        super(LSTMDQNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # LSTM for processing sequences
        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Fully connected layers after LSTM
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, action_size)
        )
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length, state_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if missing
        
        # Pass through LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the output from the last time step
        if lstm_out.dim() == 3:
            lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        q_values = self.fc(lstm_out)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state for LSTM"""
        weight = next(self.parameters())
        return (weight.new_zeros(self.lstm_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.lstm_layers, batch_size, self.hidden_size))

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class LSTMDQNAgent:
    """LSTM-DQN Agent for player behavior imitation and improvement"""
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 hidden_size=128, lstm_layers=2, sequence_length=10):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        
        # Networks
        self.policy_net = LSTMDQNetwork(state_size, action_size, hidden_size, lstm_layers).to(self.device)
        self.target_net = LSTMDQNetwork(state_size, action_size, hidden_size, lstm_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Training components
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(10000)
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Episode memory for building sequences
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Hidden state for LSTM
        self.hidden = None
    
    def reset_episode(self):
        """Reset episode memory and hidden state"""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.hidden = None
    def choose_action(self, state: np.ndarray, use_epsilon: bool = False, evaluation: bool = False):
        return self.select_action(state, evaluation)
        
    def select_action(self, state, evaluation=False):
        """Select action using epsilon-greedy policy"""
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Add to episode memory
        self.episode_states.append(state)
        
        # Create sequence from recent states
        if len(self.episode_states) >= self.sequence_length:
            state_sequence = torch.stack(self.episode_states[-self.sequence_length:])
        else:
            # Pad with zeros if not enough states
            padding = torch.zeros(self.sequence_length - len(self.episode_states), self.state_size).to(self.device)
            if self.episode_states:
                state_sequence = torch.cat([padding, torch.stack(self.episode_states)])
            else:
                state_sequence = padding
        
        state_sequence = state_sequence.unsqueeze(0)  # Add batch dimension
        
        if not evaluation and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                # Initialize hidden state if necessary
                if self.hidden is None:
                    self.hidden = self.policy_net.init_hidden(1)
                
                q_values, self.hidden = self.policy_net(state_sequence, self.hidden)
                action = q_values.max(1)[1].item()
        
        self.episode_actions.append(action)
        return action
    
    def store_transition(self, state, action, next_state, reward, done):
        """Store transition in replay buffer"""
        # Convert to tensors
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
        
        self.memory.push(state, action, next_state, reward, done)
        self.episode_rewards.append(reward)
        
        if done:
            self.reset_episode()
    
    def train(self, batch_size=64):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        experiences = self.memory.sample(batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Add sequence dimension
        state_batch = state_batch.unsqueeze(1)
        next_state_batch = next_state_batch.unsqueeze(1)
        
        # Current Q values
        current_q_values, _ = self.policy_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_batch)
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path):
        """Save the model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 0.1)
            print(f"Model loaded from {path}")
            return True
        return False
    
    def pretrain_on_player_data(self, states, actions, rewards, next_states, dones, epochs=10):
        """Pretrain the network using supervised learning on player data"""
        print("Pretraining on player data...")
        self.policy_net.train()
        
        dataset_size = len(states)
        batch_size = 64
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Get batch data
                batch_states = torch.FloatTensor(states[batch_indices]).to(self.device)
                batch_actions = torch.LongTensor(actions[batch_indices]).to(self.device)
                batch_rewards = torch.FloatTensor(rewards[batch_indices]).to(self.device)
                batch_next_states = torch.FloatTensor(next_states[batch_indices]).to(self.device)
                batch_dones = torch.FloatTensor(dones[batch_indices]).to(self.device)
                
                # Store experiences in replay buffer
                for j in range(len(batch_indices)):
                    self.memory.push(
                        batch_states[j],
                        batch_actions[j].item(),
                        batch_next_states[j],
                        batch_rewards[j].item(),
                        batch_dones[j].item()
                    )
                
                # Forward pass
                q_values, _ = self.policy_net(batch_states.unsqueeze(1))
                predicted_q_values = q_values.gather(1, batch_actions.unsqueeze(1))
                
                # Calculate target Q-values
                with torch.no_grad():
                    next_q_values, _ = self.target_net(batch_next_states.unsqueeze(1))
                    max_next_q_values = next_q_values.max(1)[0]
                    target_q_values = batch_rewards + (1 - batch_dones) * self.gamma * max_next_q_values
                
                # Compute loss
                loss = nn.MSELoss()(predicted_q_values.squeeze(), target_q_values)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Pretraining Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Update target network after pretraining
        self.update_target_network()
        print("Pretraining completed!")