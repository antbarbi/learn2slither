import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.9):
        self.q = {}             # Q-table: state -> action values
        self.actions = actions  # list of possible actions
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate

    # def get_action(self, state):
    #     # ε-greedy policy
    #     if np.random.rand() < self.epsilon or state not in self.q:
    #         return np.random.choice(self.actions)
    #     return max(self.q[state], key=self.q[state].get)

    def decay_epsilon(self, min_epsilon=0.1, decay_rate=0.999):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_action(self, state):
        if state not in self.q or np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        q_values = np.array([self.q[state][a] for a in self.actions])
        exp_q = np.exp(q_values - np.max(q_values))  # for numerical stability
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(self.actions, p=probs)

    def update(self, state, action, reward, next_state):
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q:
            self.q[next_state] = {a: 0.0 for a in self.actions}
        td_target = reward + self.gamma * max(self.q[next_state].values())
        td_error = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_error

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q = pickle.load(f)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 0.3

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.model.fc[-1].out_features)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())