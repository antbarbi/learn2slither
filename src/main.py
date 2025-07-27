import argparse
from simulator.snake import Snake, Action, GameOver
from simulator.render import draw_game
from agent import QLearningAgent
import pygame
import time
import tqdm
import multiprocessing
import glob
import pickle
import os

NUM_ROUNDS = 10
EPISODES = 100000
MAX_STEPS = 300


def load_and_merge_q_tables(pattern="q_table_worker_*.pkl", old_q_path="q_table_merged.pkl"):
    merged_q = {}
    # Load old merged Q-table if it exists
    if os.path.exists(old_q_path):
        with open(old_q_path, "rb") as f:
            old_q = pickle.load(f)
            for state, actions in old_q.items():
                merged_q[state] = actions.copy()
    # Merge worker Q-tables
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
    # Load merged Q-table if it exists
    try:
        with open("q_table_merged.pkl", "rb") as f:
            agent.q = pickle.load(f)
    except FileNotFoundError:
        pass
    max_size = 3
    for episode in tqdm.tqdm(range(episodes), desc=f"Worker {worker_id}"):
        env.reset()
        state = state_to_tuple(env.get_observation())
        no_apple_steps = 0
        for step in range(max_steps):
            action = agent.get_action(state)
            max_size = max(max_size, len(env.snake))
            try:
                env.step(action, step)
                next_state = state_to_tuple(env.get_observation())
                reward = env.get_event()
                agent.update(state, action, reward, next_state)
                state = next_state

                if any("G" in direction for direction in env.get_observation().values()):
                    no_apple_steps = 0
                else:
                    no_apple_steps += 1

                # if no_apple_steps > 30:  # Threshold for ending episode
                #     break
            except GameOver:
                reward = env.get_event()
                agent.update(state, action, reward, state)
                break
        agent.decay_epsilon()
    agent.save_q_table(f"q_table_worker_{worker_id}.pkl")
    print(f"Worker {worker_id} max size: {max_size}")


def main(render=True, workers=4):
    if render:
        env = Snake()
        agent = QLearningAgent(actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT], epsilon=0)
        max_size = 3
        try:
            agent.load_q_table("q_table_merged.pkl")
        except Exception:
            pass

        def state_to_tuple(state_dict):
            return tuple(tuple(v) for v in state_dict.values())

        pygame.init()
        from simulator.render import WINDOW_SIZE
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Learn2Slither")

        episode_iter = range(EPISODES)
        env.print_observation()
        for episode in episode_iter:
            env.reset()
            state = state_to_tuple(env.get_observation())
            for step in range(MAX_STEPS):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                action = agent.get_action(state)
                max_size = max(max_size, len(env.snake))
                try:
                    env.step(action, step)
                    next_state = state_to_tuple(env.get_observation())
                    reward = env.get_event()
                    # agent.update(state, action, reward, next_state)
                    state = next_state
                except GameOver as e:
                    reward = env.get_event()
                    agent.update(state, action, reward, state)
                    with open("error.txt", "a") as error_file:  # Use "a" to append
                        error_file.write(f"Episode {episode}, Step {step}: {str(e)}\n")
                    break  # End episode on GameOver

                draw_game(screen, env)
                pygame.display.flip()
                time.sleep(0.1)
                print(action)
                env.print_observation()
                print(step, ":", env.total_reward)

        agent.save_q_table("q_table.pkl")
        pygame.quit()
        print(max_size)
    else:
        num_workers = workers
        episodes_per_worker = EPISODES // (num_workers // 1)
        for round in range(NUM_ROUNDS):
            processes = []
            for i in range(num_workers):
                p = multiprocessing.Process(target=train_worker, args=(i, episodes_per_worker, MAX_STEPS))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (non-render mode)")
    args = parser.parse_args()
    main(render=args.render, workers=args.workers)
