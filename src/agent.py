import numpy as np

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q = {}  # Q-table: state -> action values
        self.actions = actions  # list of possible actions
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate

    def get_action(self, state):
        # ε-greedy policy
        if np.random.rand() < self.epsilon or state not in self.q:
            return np.random.choice(self.actions)
        return max(self.q[state], key=self.q[state].get)

    def update(self, state, action, reward, next_state):
        if state not in self.q:
            self.q[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q:
            self.q[next_state] = {a: 0.0 for a in self.actions}
        td_target = reward + self.gamma * max(self.q[next_state].values())
        td_error = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_error