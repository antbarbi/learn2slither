import multiprocessing
import pickle
import glob
import os
import tqdm
from simulator.snake import Snake, Action, GameOver
from agent import QLearningAgent


NUM_ROUNDS = 100
DEFAULT_EPISODES = 100000
DEFAULT_MAX_STEPS = 300

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
            max_size = max(episode_max_size, len(env.snake))
            
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


def train_main(workers=2, episodes=DEFAULT_EPISODES, steps=DEFAULT_MAX_STEPS):
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
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--episodes", "-e", type=int, default=DEFAULT_EPISODES, help="Number of episodes")
    parser.add_argument("--steps", "-s", type=int, default=DEFAULT_MAX_STEPS, help="Max steps per episode")
    args = parser.parse_args()
    train_main(workers=args.workers, episodes=args.episodes, steps=args.steps)
