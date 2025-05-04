# agents.py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import os
import threading
import gc

from models import DQN
from config import *

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_CAPACITY):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Ensure we're storing copies to avoid memory issues
        self.buffer.append((
            np.array(state, copy=True),
            action,
            float(reward),
            np.array(next_state, copy=True),
            float(done)
        ))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.target_update = TARGET_UPDATE
        self.train_step = 0
        
        # Use CPU to avoid GPU memory issues
        self.device = torch.device("cpu")
        
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.episodes = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Load model if exists
        self.load_model()
    
    def choose_action(self, state):
        with self.lock:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        with self.lock:
            try:
                states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
                
                # Convert to tensors
                states = torch.FloatTensor(states).to(self.device)
                actions = torch.LongTensor(actions).to(self.device)
                rewards = torch.FloatTensor(rewards).to(self.device)
                next_states = torch.FloatTensor(next_states).to(self.device)
                dones = torch.FloatTensor(dones).to(self.device)
                
                # Get current Q values
                current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
                
                # Get next Q values
                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1)[0]
                    expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
                
                # Compute loss
                loss = F.mse_loss(current_q_values.squeeze(), expected_q_values)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()
                
                # Update target network
                self.train_step += 1
                if self.train_step % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Clear some memory
                del states, actions, rewards, next_states, dones
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
            except Exception as e:
                print(f"Error in training: {e}")
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filename='dqn_model.pth'):
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'episodes': self.episodes,
                'train_step': self.train_step
            }, filename)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filename='dqn_model.pth'):
        if os.path.exists(filename):
            try:
                checkpoint = torch.load(filename, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', 1.0)
                self.episodes = checkpoint.get('episodes', 0)
                self.train_step = checkpoint.get('train_step', 0)
                print(f"Model loaded. Episodes: {self.episodes}, Epsilon: {self.epsilon}")
            except Exception as e:
                print(f"Error loading model: {e}")