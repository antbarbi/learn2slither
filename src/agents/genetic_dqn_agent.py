from .dqn_agent import DQNAgent
import torch
import numpy as np
import os
from simulator.snake import Snake, GameOver, Action
import random
from joblib import Parallel, delayed
from collections import deque

class GlobalReplayBuffer:
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experiences):
        self.buffer.extend(experiences)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

def flatten_state(state_list):
    """Convert state list to numpy array for DQN input"""
    return np.array([item['value'] for item in state_list], dtype=np.float32)


def train_single_agent(agent: DQNAgent, episodes: int, max_steps=300):
    for episode in range(episodes):
        env = Snake()
        env.reset()
        state = flatten_state(env.get_state())
        for _ in range(max_steps):
            action = agent.get_action(state)
            try:
                env.step(Action(action))
                next_state = flatten_state(env.get_state())
                agent.remember(state, action, env.reward, next_state, False)
                agent.train()
                state = next_state
            except GameOver:
                agent.remember(state, action, env.reward, state, True)
                agent.train()
                break
        if episode % 100 == 0:
            agent.update_target()
        agent.decay_epsilon()

def eval_single_agent(agent: DQNAgent, episodes: int):
    total_reward = 0
    total_steps = 0
    total_size = 0
    
    original_epsilon = agent.epsilon  # ← FIX: Save original epsilon
    agent.epsilon = 0.0  # No exploration during evaluation
    
    for _ in range(episodes):
        env = Snake()
        env.reset()
        state = flatten_state(env.get_state())
        done = False
        steps = 0
        max_size = len(env.snake)
        
        while not done and steps < 500:  # Max 500 steps per episode
            action = agent.get_action(state)
            try:
                env.step(Action(action))
                state = flatten_state(env.get_state())
                total_reward += env.reward
                max_size = max(max_size, len(env.snake))
                steps += 1
            except GameOver:
                done = True
        
    #     total_steps += steps
    #     total_size += max_size
    
    agent.epsilon = original_epsilon  # ← FIX: Restore epsilon
    # # Composite fitness: reward + survival + size
    # avg_reward = total_reward / episodes
    # avg_steps = total_steps / episodes
    # avg_size = total_size / episodes
    
    # fitness = avg_reward + (avg_steps * 0.1) + (avg_size * 5)
    # return fitness

    return total_reward / episodes

class GeneticDQNAgent:
    def __init__(self, state_dim, action_dim, population_size=10, base_model_path=None):
        self.global_replay_buffer = GlobalReplayBuffer()
        self.population_size = population_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population = []
        self.fitness = [0] * population_size
        self.generation = 0  # Track current generation
        
        # Initialize population
        if base_model_path and os.path.exists(base_model_path):
            # Load the base model and create population from it
            base_agent = DQNAgent(self.state_dim, self.action_dim)
            base_agent.load_model(base_model_path)
            
            self.population = []
            for i in range(self.population_size):
                agent = DQNAgent(self.state_dim, self.action_dim)
                agent.load_model(base_model_path)
                # Add small random variations to create genetic diversity
                if i > 0:  # Keep first agent as exact copy
                    self.mutate(agent)
                self.population.append(agent)
        else:
            # Create random population
            self.population = [DQNAgent(self.state_dim, self.action_dim) for _ in range(self.population_size)]

    def collect_experiences(self):
        for agent in self.population:
            self.global_replay_buffer.add(agent.memory)  # assuming agent.memory is a list
            agent.memory.clear()  # Optional: clear agent memory after collection

    def initialize_agent_memory(self, agent: DQNAgent, batch_size=1000):
        if len(self.global_replay_buffer.buffer) >= batch_size:
            agent.memory = list(self.global_replay_buffer.sample(batch_size))

    def evaluate_population(self, episodes=50, n_jobs=-2):
        """Evaluate all agents in parallel"""
        self.fitness = Parallel(n_jobs=n_jobs)(
            delayed(eval_single_agent)(agent, episodes)
            for agent in self.population
        )

    def train_population(self, episodes=100, n_jobs=-2):
        """Train all agents in parallel"""
        Parallel(n_jobs=n_jobs)(
            delayed(train_single_agent)(agent, episodes)
            for agent in self.population
        )

    def select_elite(self, elite_fraction=0.1):
        elite_count = max(1, int(self.population_size * elite_fraction))
        elite_indices = np.argsort(self.fitness)[-elite_count:]
        return [self.population[i] for i in elite_indices]

    def crossover(self, parent1, parent2):
        child = DQNAgent(self.state_dim, self.action_dim)
        for child_param, p1_param, p2_param in zip(child.model.parameters(), parent1.model.parameters(), parent2.model.parameters()):
            mask = torch.rand_like(child_param) < 0.5
            child_param.data.copy_(torch.where(mask, p1_param.data, p2_param.data))
        return child

    def mutate(self, agent, mutation_rate=0.1):
        for param in agent.model.parameters():
            param.data += torch.randn_like(param) * mutation_rate

    def _copy_agent(self, agent):
        """Create a deep copy of an agent"""
        new_agent = DQNAgent(self.state_dim, self.action_dim)
        new_agent.model.load_state_dict(agent.model.state_dict())
        new_agent.target_model.load_state_dict(agent.target_model.state_dict())
        new_agent.epsilon = agent.epsilon
        new_agent.epsilon_min = agent.epsilon_min
        new_agent.epsilon_decay = agent.epsilon_decay
        return new_agent

    def evolve(self, random_injection=0.5):
        elites = self.select_elite()
        new_population = []
        
        # Keep elites
        for elite in elites:
            new_elite = self._copy_agent(elite)
            new_population.append(new_elite)
        
        # Create children
        while len(new_population) < self.population_size:
            if random.random() < random_injection:
                child = DQNAgent(self.state_dim, self.action_dim)
            else:
                if len(elites) >= 2:
                    parents = random.sample(elites, 2)
                else:
                    # If only one elite, use it twice
                    parents = [elites[0], elites[0]]
                child = self.crossover(parents[0], parents[1])
                self.mutate(child)
            child.epsilon = max(0.05, 0.9 * (0.95 ** self.generation))
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1  # Increment generation counter

        # Initialize replay buffer for each agent from global buffer
        for agent in self.population:
            self.initialize_agent_memory(agent, batch_size=5000)