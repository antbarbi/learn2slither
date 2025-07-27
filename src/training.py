import multiprocessing
import pickle
import glob
import os
import tqdm
from simulator.snake import Snake, Action, GameOver
from agent import QLearningAgent
import numpy as np
import torch

DEFAULT_WORKERS = 2
NUM_ROUNDS = 100
DEFAULT_EPISODES = 100000
DEFAULT_MAX_STEPS = 300

def flatten_state(state_dict):
    # Flattens the vision dict into a numeric vector for DQN
    mapping = {'0': 0, 'W': 1, 'S': 2, 'G': 3, 'R': 4}
    flat = []
    for direction in ['up', 'down', 'left', 'right']:
        for cell in state_dict[direction]:
            flat.append(mapping.get(str(cell), 0))
    return np.array(flat, dtype=np.float32)


def train_DQNAgent(worker_id, episodes, max_steps, state_dim, action_dim):
    from agent import DQNAgent
    env = Snake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim)
    agent.model.to(device)
    agent.target_model.to(device)
    max_size = 3
    total_size = 0
    total_reward = 0
    for episode in tqdm.tqdm(range(episodes), desc=f"Worker {worker_id}"):
        env.reset()
        state = flatten_state(env.get_observation())
        state = torch.FloatTensor(state).to(device)
        episode_max_size = len(env.snake)
        episode_total_reward = 0
        for step in range(max_steps):
            action_idx = agent.get_action(state.cpu().numpy())
            action = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT][action_idx]
            episode_max_size = max(episode_max_size, len(env.snake))
            max_size = max(max_size, len(env.snake))
            try:
                env.step(action, step)
                next_state = flatten_state(env.get_observation())
                next_state = torch.FloatTensor(next_state).to(device)
                reward = env.get_reward()
                done = False
                episode_total_reward += reward
                agent.remember(state.cpu().numpy(), action_idx, reward, next_state.cpu().numpy(), done)
                agent.train()
                state = next_state
            except GameOver:
                reward = env.get_reward()
                done = True
                episode_total_reward += reward
                agent.remember(state.cpu().numpy(), action_idx, reward, state.cpu().numpy(), done)
                agent.train()
                break
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        agent.update_target()
        total_size += episode_max_size
        total_reward += episode_total_reward
    avg_size = total_size / episodes
    avg_reward = total_reward / episodes
    print(f"Worker {worker_id} max size: {max_size:.2f}")
    print(f"Worker {worker_id} average max size: {avg_size:.2f}")
    print(f"Worker {worker_id} average total reward: {avg_reward:.2f}")
    torch.save(agent.model.state_dict(), f"dqn_weights_worker_{worker_id}.pt")

def train_DQN(workers=DEFAULT_WORKERS, episodes=DEFAULT_EPISODES, steps=DEFAULT_MAX_STEPS):
    from agent import DQNAgent
    env = Snake()
    state_dim = len(flatten_state(env.get_observation()))
    action_dim = 4
    num_workers = workers
    episodes_per_worker = episodes // num_workers
    for round in range(NUM_ROUNDS):
        processes = []
        for i in range(num_workers):
            p = multiprocessing.Process(target=train_DQNAgent, args=(i, episodes_per_worker, steps, state_dim, action_dim))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print("DQN weights saved for all workers.")


#### Qlearning

def load_and_merge_q_tables(pattern="q_table_worker_*.pkl", old_q_path="q_table_merged.pkl"):
    merged_q = {}
    if os.path.exists(old_q_path):
        with open(old_q_path, "rb") as f:
            old_q = pickle.load(f)
            for state, actions in old_q.items():
                merged_q[state] = actions.copy()
    for filename in glob.glob(pattern):
        with open(filename, "rb") as f:
            worker_q = pickle.load(f)
            for state, actions in worker_q.items():
                if state not in merged_q:
                    merged_q[state] = actions.copy()
                else:
                    for action, value in actions.items():
                        if action in merged_q[state]:
                            merged_q[state][action] = max(merged_q[state][action], value)
                        else:
                            merged_q[state][action] = value
    return merged_q

def state_to_tuple(state_dict):
    return tuple(tuple(v) for v in state_dict.values())

def train_worker(worker_id, episodes, max_steps):
    env = Snake()
    agent = QLearningAgent(actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])
    
    try:
        with open("q_table_merged.pkl", "rb") as f:
            agent.q = pickle.load(f)
    except FileNotFoundError:
        pass
 
    max_size = 3
    total_size = 0
    total_reward = 0
    for episode in tqdm.tqdm(range(episodes), desc=f"Worker {worker_id}"):
        env.reset()
        state = state_to_tuple(env.get_observation())
        episode_max_size = len(env.snake)
        episode_total_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            episode_max_size = max(episode_max_size, len(env.snake))
            max_size = max(max_size, len(env.snake))
            
            try:
                env.step(action, step)
                next_state = state_to_tuple(env.get_observation())
                reward = env.get_reward()
                episode_total_reward += reward
                agent.update(state, action, reward, next_state)
                state = next_state
            except GameOver:
                reward = env.get_reward()
                episode_total_reward += reward
                agent.update(state, action, reward, state)
                break
        
        agent.decay_epsilon()
        total_size += episode_max_size
        total_reward += episode_total_reward
    
    avg_size = total_size / episodes
    avg_reward = total_reward / episodes
    print(f"Worker {worker_id} max size: {max_size:.2f}")
    print(f"Worker {worker_id} average max size: {avg_size:.2f}")
    print(f"Worker {worker_id} average total reward: {avg_reward:.2f}")
    agent.save_q_table(f"q_table_worker_{worker_id}.pkl")


def train_QLearningAgent(workers=DEFAULT_WORKERS, episodes=DEFAULT_EPISODES, steps=DEFAULT_MAX_STEPS):
    num_workers = workers
    episodes_per_worker = episodes // num_workers
    for round in range(NUM_ROUNDS):
        processes = []
        for i in range(num_workers):
            p = multiprocessing.Process(target=train_worker, args=(i, episodes_per_worker, steps))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        merged_q = load_and_merge_q_tables()
        print("Starting to save Q table.")
        with open("q_table_merged.pkl", "wb") as f:
            pickle.dump(merged_q, f)
        print("Q table saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel workers")
    parser.add_argument("--episodes", "-e", type=int, default=DEFAULT_EPISODES, help="Number of episodes")
    parser.add_argument("--steps", "-s", type=int, default=DEFAULT_MAX_STEPS, help="Max steps per episode")
    parser.add_argument("--agent", choices=["q", "dqn"], default="q", help="Choose agent type: qlearning or dqn")
    args = parser.parse_args()

    if args.agent == "q":
        train_QLearningAgent(workers=args.workers, episodes=args.episodes, steps=args.steps)
    elif args.agent == "dqn":
        train_DQN(workers=args.workers, episodes=args.episodes, steps=args.steps)
