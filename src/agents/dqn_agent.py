import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from collections import deque
import random
import os


def get_snake_std(snake):
    positions = np.array(snake)
    std_x = np.std(positions[:, 0])
    std_y = np.std(positions[:, 1])
    return (std_x + std_y) / 2


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, state_dim * 6),
            nn.LayerNorm(state_dim * 6),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(state_dim * 6, state_dim * 3),
            nn.LayerNorm(state_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(state_dim * 3, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.00005, gamma=0.99, batch_size=128, memory_size=10000):
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.epsilon = 0.5
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.loss_fn = nn.MSELoss()
        self.scaler = GradScaler()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.model.fc[-1].out_features)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        with autocast(device_type=str(self.device)):
            q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (~dones)
            loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        if not os.path.exists(path):
            return False

        loaded = torch.load(path, map_location=self.device)
        # Try strict load first
        try:
            self.model.load_state_dict(loaded)
            self.update_target()
            return True
        except (RuntimeError, ValueError) as e:
            # Fallback: copy only tensors whose shapes match
            model_sd = self.model.state_dict()
            to_copy = {}
            for k, v in loaded.items():
                if k in model_sd and isinstance(v, torch.Tensor) and v.size() == model_sd[k].size():
                    to_copy[k] = v

            if not to_copy:
                # Nothing compatible to copy
                raise e

            model_sd.update(to_copy)
            self.model.load_state_dict(model_sd)
            self.update_target()
            print(f"Warning: partially loaded model from {path} — skipped {len(loaded) - len(to_copy)} tensors due to shape mismatch.")
            return True

