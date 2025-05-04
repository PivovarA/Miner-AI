# player_behavior_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# Define actions to match the game
ACTIONS = ['move_up', 'move_down', 'move_left', 'move_right', 'drop_trap', 'explode_traps', 'wait']

class PlayerDataset(Dataset):
    """Dataset for player behavior training"""
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        
        # Convert string actions to indices
        self.action_indices = []
        for action in actions:
            if isinstance(action, str):
                self.action_indices.append(ACTIONS.index(action))
            else:
                self.action_indices.append(action)
        
        self.actions = torch.LongTensor(self.action_indices)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class PlayerBehaviorNetwork(nn.Module):
    """Neural network for imitating player behavior"""
    def __init__(self, state_size, action_size):
        super(PlayerBehaviorNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class PlayerBehaviorAgent:
    """Agent that imitates player behavior"""
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = PlayerBehaviorNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # For exploration (if needed)
        self.epsilon = 0.1
    
    def choose_action(self, state, use_epsilon=True):
        """Choose an action based on the current state"""
        # Random action with epsilon probability
        if use_epsilon and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Otherwise, use the model
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            elif not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(np.array(state)).to(self.device)
            
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            logits = self.model(state)
            action = torch.argmax(logits, dim=1).item()
            return action
    
    def train_supervised(self, states, actions, batch_size=64, epochs=10):
        """Train the model using supervised learning"""
        dataset = PlayerDataset(states, actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_states, batch_actions in dataloader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)
                
                # Forward pass
                logits = self.model(batch_states)
                loss = self.criterion(logits, batch_actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_actions).sum().item()
                total += batch_states.size(0)
            
            accuracy = correct / total
            avg_loss = total_loss / len(dataloader)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def save_model(self, path):
        """Save the model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model from disk"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 0.1)
            print(f"Model loaded from {path}")
            return True
        return False