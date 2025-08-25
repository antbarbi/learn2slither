import numpy as np
from simulator.feature_engineering import SnakeFeatureEngineering, Snake
import os
# hide pygame support prompt when importing pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


DEFAULT_ROUNDS = 1
DEFAULT_EPISODES = 2000
DEFAULT_MAX_STEPS = 300

# Genetic DQN support removed — training now uses standard DQN only


def train_DQN(
        episodes=DEFAULT_EPISODES,
        steps=DEFAULT_MAX_STEPS,
        rounds=DEFAULT_ROUNDS,
        state_type: str = "base",
        reward_type: str = "base",
        history_k: int = 1):
    """Train using standard DQN"""
    os.makedirs("models", exist_ok=True)

    # Get state dimensions from environment using selected feature engineering
    env = Snake()
    env.reset()
    fe = SnakeFeatureEngineering(
        state_type=state_type,
        reward_type=reward_type,
        history_k=history_k)
    fe.reset_history(env)
    print(fe.extract_state(env))
    state_dim = len(fe.extract_state(env))
    action_dim = 3

    print(
        f"State dimension: {state_dim}",
        f"Action dimension: {action_dim}",
        f"history: {fe.history_k}"
    )

    # Import DQNAgent here to avoid circular imports
    from agents.dqn_agent import DQNAgent

    # Create DQN agent
    agent = DQNAgent(state_dim, action_dim)

    # Training statistics
    episode_rewards = []
    best_reward = float('-inf')

    for round_num in range(rounds):
        print(f"\n=== Round {round_num + 1}/{rounds} ===")

        if os.path.exists("models/dqn_best.pt"):
            print("Loading best model from previous round...")
            agent.load_model("models/dqn_best.pt")
            print("Best model loaded successfully!")

        for episode in range(episodes):
            env.reset()
            fe.reset_history(env)
            state = fe.extract_state(env)
            episode_reward = 0
            steps_taken = 0
            fe.total_reward = 0

            for step in range(steps):
                # Agent returns a relative action index:
                # 0=forward,1=left,2=right
                rel_action = int(agent.get_action(state))
                # Map relative -> absolute action before stepping the env
                abs_action = fe.relative_to_action(env, rel_action)
                reward, done = fe.step_and_compute_reward(
                    env, abs_action)   # <- returns (reward, done)
                next_state = fe.extract_state(env)
                episode_reward += reward
                steps_taken += 1

                # Store relative action index in replay buffer
                agent.remember(state, rel_action, reward, next_state, done)
                agent.train()

                state = next_state
                if done:
                    break
            # Update target network periodically
            if episode % 100 == 0:
                agent.update_target()

            # Decay epsilon
            agent.decay_epsilon()

            # Track rewards
            episode_rewards.append(episode_reward)

            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Snake Size: {len(env.snake)}, Steps: {steps_taken}, "
                      f"Epsilon: {agent.epsilon:.3f}")

                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    agent.save_model("models/dqn_best.pt")
                    print(
                        f"New best average reward: {best_reward:.2f}",
                        "- Model saved!"
                    )

        # Save model after each round
        agent.save_model(f"models/dqn_round_{round_num + 1}.pt")
        print(f"Round {round_num + 1} completed. Model saved.")

    # Save final model
    agent.save_model("models/dqn_final.pt")

    print("\nTraining completed!")
    print(f"Best average reward: {best_reward:.2f}")
    print("Final model saved as 'models/dqn_final.pt'")
    print("Best model saved as 'models/dqn_best.pt'")

    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-e", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--steps", "-s", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--rounds", "-r", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument(
        "--history-k",
        type=int,
        default=1,
        help="How many historical frames to include \
            in the flattened state (history_k). Default=1"
    )
    # (genetic training removed)
    # Feature engineering selection
    parser.add_argument(
        "--state-fn",
        choices=list(
            SnakeFeatureEngineering.STATE_FUNCTIONS.keys()),
        default="base",
        help="Which state function to use from simulator.feature_engineering")
    parser.add_argument(
        "--reward-fn",
        choices=list(
            SnakeFeatureEngineering.REWARD_FUNCTIONS.keys()),
        default="base",
        help="Which reward function to use from simulator.feature_engineering")

    args = parser.parse_args()

    train_DQN(
        episodes=args.episodes,
        steps=args.steps,
        rounds=args.rounds,
        state_type=args.state_fn,
        reward_type=args.reward_fn,
        history_k=args.history_k)
