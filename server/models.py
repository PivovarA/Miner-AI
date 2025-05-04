# models.py
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class LSTMDQNetwork(nn.Module):
    """LSTM-based Deep Q-Network for handling sequential game states"""
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