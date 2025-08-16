import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Add this line FIRST

from simulator.feature_engineering import SnakeFeatureEngineering, Snake, GameOver, Action
from agents.genetic_dqn_agent import GeneticDQNAgent
import numpy as np

DEFAULT_ROUNDS = 1
DEFAULT_EPISODES = 2000  # Reduced for faster testing
DEFAULT_MAX_STEPS = 300

def flatten_state(state_list):
    """Convert state list to numpy array for DQN input"""
    return np.array([item['value'] for item in state_list], dtype=np.float32)

def train_genetic_DQN(generations=50, population_size=20, episodes_per_generation=100, evaluation_episodes=20,
                      state_type: str = "base", reward_type: str = "base"):
    """Train using Genetic Algorithm + DQN"""
    os.makedirs("models", exist_ok=True)
    
    # Get state dimensions from a dummy environment using the requested feature engineering
    env = Snake()
    env.reset()
    fe = SnakeFeatureEngineering(state_type=state_type, reward_type=reward_type)
    state_dim = len(fe.extract_state(env))  # dynamic state size based on chosen state_fn
    action_dim = 4
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create genetic DQN agent
    genetic_agent = GeneticDQNAgent(
        state_dim,
        action_dim,
        population_size=population_size,
        base_model_path="./dqn_best.pt"
    )
    
    best_fitness_history = []
    
    for generation in range(generations):
        print(f"\n=== Generation {generation + 1}/{generations} ===", flush=True)
        
        # Train population
        print("Training population...")
        genetic_agent.train_population(episodes=episodes_per_generation, n_jobs=-2)
        
        # Evaluate population
        print("Evaluating population...")
        genetic_agent.evaluate_population(episodes=evaluation_episodes, n_jobs=-2)
        
        # Print fitness statistics
        fitness_scores = genetic_agent.fitness
        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        worst_fitness = min(fitness_scores)
        
        print(f"Generation {generation + 1} Results:")
        print(f"  Best Fitness: {best_fitness:.2f}")
        print(f"  Average Fitness: {avg_fitness:.2f}")
        print(f"  Worst Fitness: {worst_fitness:.2f}")
        print(f"  Current Epsilon: {genetic_agent.population[0].epsilon:.3f}")
        print(f"  Generation: {genetic_agent.generation}")
        
        best_fitness_history.append(best_fitness)
        
        # Save best agent from this generation
        best_agent_idx = np.argmax(fitness_scores)
        best_agent = genetic_agent.population[best_agent_idx]
        best_agent.save_model(f"models/genetic_best_gen_{generation + 1}.pt")
        
        # Evolve to next generation
        print("Evolving to next generation...")
        genetic_agent.evolve(random_injection=0.1)
        
        # Early stopping if converged
        if generation >= 5:
            recent_best = best_fitness_history[-5:]
            if max(recent_best) - min(recent_best) < 1.0:
                print("Population converged! Stopping early.")
                break
    
    # Save final best agent
    genetic_agent.evaluate_population(episodes=evaluation_episodes, n_jobs=-2)
    final_best_idx = np.argmax(genetic_agent.fitness)
    final_best_agent = genetic_agent.population[final_best_idx]
    final_best_agent.save_model("models/genetic_final_best.pt")
    
    print(f"\nTraining completed! Best fitness: {max(genetic_agent.fitness):.2f}")
    print("Final best model saved as 'models/genetic_final_best.pt'")
    
    return final_best_agent

def train_DQN(episodes=DEFAULT_EPISODES, steps=DEFAULT_MAX_STEPS, rounds=DEFAULT_ROUNDS,
              state_type: str = "base", reward_type: str = "base"):
    """Train using standard DQN"""
    os.makedirs("models", exist_ok=True)
    
    # Get state dimensions from environment using selected feature engineering
    env = Snake()
    env.reset()
    fe = SnakeFeatureEngineering(state_type=state_type, reward_type=reward_type)
    print(fe.extract_state(env))
    state_dim = len(fe.extract_state(env))
    action_dim = 4
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
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
            state = fe.extract_state(env)
            episode_reward = 0
            steps_taken = 0
            fe.total_reward = 0
            
            for step in range(steps):
                
                action = agent.get_action(state)
                reward, done = fe.step_and_compute_reward(env, action)   # <- returns (reward, done)
                next_state = fe.extract_state(env)
                episode_reward += reward
                steps_taken += 1

                agent.remember(state, action, reward, next_state, done)
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
                    print(f"New best average reward: {best_reward:.2f} - Model saved!")
        
        # Save model after each round
        agent.save_model(f"models/dqn_round_{round_num + 1}.pt")
        print(f"Round {round_num + 1} completed. Model saved.")
    
    # Save final model
    agent.save_model("models/dqn_final.pt")
    
    print(f"\nTraining completed!")
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
    parser.add_argument("--genetic", action="store_true", help="Use genetic DQN training")
    parser.add_argument("--generations", "-g", type=int, default=50, help="Number of generations for genetic training")
    parser.add_argument("--population", "-p", type=int, default=10, help="Population size for genetic training")
    # Feature engineering selection
    parser.add_argument("--state-fn", choices=list(SnakeFeatureEngineering.STATE_FUNCTIONS.keys()),
                        default="base", help="Which state function to use from simulator.feature_engineering")
    parser.add_argument("--reward-fn", choices=list(SnakeFeatureEngineering.REWARD_FUNCTIONS.keys()),
                        default="base", help="Which reward function to use from simulator.feature_engineering")
    
    args = parser.parse_args()
    
    if args.genetic:
        train_genetic_DQN(
            generations=args.generations,
            population_size=args.population,
            episodes_per_generation=200,
            evaluation_episodes=20,
            state_type=args.state_fn,
            reward_type=args.reward_fn,
        )
    else:
        train_DQN(episodes=args.episodes, steps=args.steps, rounds=args.rounds,
                  state_type=args.state_fn, reward_type=args.reward_fn)