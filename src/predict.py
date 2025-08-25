import argparse
import pygame
import time
from simulator.render import WINDOW_SIZE, draw_game
from simulator.feature_engineering import Snake, SnakeFeatureEngineering
from agents.dqn_agent import DQNAgent


def main(
        weights: str,
        episodes: int,
        steps: int,
        state_fn: str = "base",
        reward_fn: str = "base",
        history_k: int = 1):
    env = Snake()

    # Instantiate feature engineering chosen by CLI
    env.reset()
    fe = SnakeFeatureEngineering(
        state_type=state_fn,
        reward_type=reward_fn,
        history_k=history_k)
    fe.reset_history(env)
    # dynamic features depending on state function
    state_dim = len(fe.extract_state(env))
    action_dim = 3

    print(
        f"State dimension: {state_dim}, \
        Action dimension: {action_dim}, \
        state_fn: {state_fn}, \
        reward_fn: {reward_fn}"
    )

    # Create DQN agent with no exploration (epsilon=0 for inference)
    agent = DQNAgent(state_dim, action_dim)
    agent.epsilon = 0  # No exploration during prediction

    max_snake_size = 0
    try:
        print("Loading DQN model...")
        success = agent.load_model(weights)
        if not success:
            print("Failed to load model. Please check the file path.")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you're using a valid .pt model\
             file from genetic DQN training.")
        return
    print("DQN model loaded successfully!")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Learn2Slither - Genetic DQN Agent")

    for episode in range(episodes):
        env.reset()
        fe.reset_history(env)
        episode_reward = 0

        for step in range(steps):
            # Use get_state() instead of get_observation()
            state = fe.extract_state(env)
            max_snake_size = max(max_snake_size, len(env.snake))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Get action from DQN agent
            # 0/1/2 relative
            rel_action = int(agent.get_action(state))
            abs_action = fe.relative_to_action(
                env, rel_action)          # map to absolute Action
            # Step the environment using the selected reward function
            reward, done = fe.step_and_compute_reward(env, abs_action)
            # debug
            print(f"rel={rel_action} -> abs={abs_action.name}")
            episode_reward += reward
            if done:
                break

            draw_game(screen, env)
            pygame.display.flip()
            time.sleep(0.1)

            env.print_observation()
            print(
                f"Snake size: {len(env.snake)}, "
                f"Best size: {max_snake_size}, "
                f"Step: {step}, "
                f"Episode reward: {episode_reward:.1f}, "
            )

        print(
            f"Episode {episode + 1} finished. \
            Final reward: {episode_reward:.1f}"
        )

    pygame.quit()
    print(
        f"All episodes completed. Best snake size achieved: {max_snake_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Path to genetic DQN model file (.pt)")
    parser.add_argument("--episodes", "-e", type=int, default=5)
    parser.add_argument("--steps", "-s", type=int, default=500)
    parser.add_argument(
        "--state-fn",
        choices=list(
            SnakeFeatureEngineering.STATE_FUNCTIONS.keys()),
        default="base",
        help="Which state function to use for feature extraction")
    parser.add_argument(
        "--reward-fn",
        choices=list(
            SnakeFeatureEngineering.REWARD_FUNCTIONS.keys()),
        default="base",
        help="Which reward function to use during prediction")
    parser.add_argument(
        "--history-k",
        type=int,
        default=1,
        help="How many historical frames to include in"
        "the flattened state (history_k). Default=1"
        )
    args = parser.parse_args()
    main(
        args.weights,
        args.episodes,
        args.steps,
        state_fn=args.state_fn,
        reward_fn=args.reward_fn,
        history_k=args.history_k)
