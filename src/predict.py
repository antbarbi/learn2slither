import argparse
import pygame
import time
from simulator.render import WINDOW_SIZE, draw_game
from simulator.snake import Snake, GameOver, Action
from agent import QLearningAgent


def state_to_tuple(state_dict):
    return tuple(tuple(v) for v in state_dict.values())


def main(weights: str, episodes: int, steps: int):
    env = Snake()
    agent = QLearningAgent(actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT], epsilon=0)
    max_snake_size = 0
    try:
        print("Loading Q table...")
        agent.load_q_table(weights)
    except Exception:
        print("No weights detected")
        return
    print("Q table loaded !")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Learn2Slither")
    for episode in range(episodes):
        env.reset()
        for step in range(steps):
            state = state_to_tuple(env.get_observation())
            max_snake_size = max(max_snake_size, len(env.snake))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            action = agent.get_action(state)
            
            try:
                env.step(action, step)
            except GameOver as e:
                with open("error.txt", "a") as error_file:  # Use "a" to append
                    error_file.write(f"Episode {episode}, Step {step}: {str(e)}\n")
                break

            draw_game(screen, env)
            pygame.display.flip()
            time.sleep(0.1)
            print(action)
            env.print_observation()
            print(
                "Actual size :", len(env.snake),
                "Best size :", max_snake_size,
                "step :",step,
                "reward :", env.total_reward
            )

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Path to weights file") 
    parser.add_argument("--episodes", "-e", type=int, default=50)
    parser.add_argument("--steps", "-s", type=int, default=300)
    args = parser.parse_args()
    main(args.weights ,args.episodes, args.steps)