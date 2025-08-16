# Q learning

## Quality function

$$
Q(s, a) = \text{quality of state/action pair}
$$

$$
Q(s, a) = \mathbb{E} \left[ R(s', s, a) + \gamma V(s') \right]
$$

$$
Q(s, a) = \sum_{s'} P(s' | s, a) \left[ R(s', s, a) + \gamma V(s') \right]
$$

Where $Q(s, a)$ is the expected future reward, given that I am in state $s$ now and take action $a$.

## Value Function and Policy

The value function $V(s)$ gives the maximum expected reward achievable from state $s$ by choosing the best action:

$$
V(s) = \max_a Q(s, a)
$$

The optimal policy $\pi(s, a)$ selects the action $a$ that maximizes the quality function $Q(s, a)$:

$$
\pi(s, a) = \operatorname{argmax}_a Q(s, a)
$$


## Monte Carlo Learning

Monte Carlo methods estimate the value and quality functions by averaging over complete episodes. The total reward over an episode is:

$$
R_\Sigma = \sum_{k=1}^n \gamma^k r_k
$$

Where $r_k$ is the reward at time step $k$, and $\gamma$ is the discount factor.

The value function is updated as:

$$
v^{\text{new}}(s_k) = v^{\text{old}}(s_k) + \frac{1}{n} \left( R_\Sigma - v^{\text{old}}(s_k) \right) \quad \forall k \in [1, \ldots, n]
$$

The quality function is updated as:

$$
Q^{\text{new}}(s_k, a_k) = Q^{\text{old}}(s_k, a_k) + \frac{1}{n} \left( R_\Sigma - Q^{\text{old}}(s_k, a_k) \right) \quad \forall k \in [1, \ldots, n]
$$

**Explanation:**
- $R_\Sigma$ is the total discounted reward collected over an episode.
- The value and quality functions are updated by moving their old values towards the observed total reward, averaged over the episode length.
- This approach requires waiting until the end of an episode before updating values.

In summary:

## Model-Based vs Model-Free Methods

**Model-based** reinforcement learning methods use a model of the environment (i.e., they know or learn $P(s'|s,a)$ and $R(s',s,a)$) to plan and make decisions. These methods can simulate future states and rewards to improve learning and decision-making.

**Model-free** methods do not use or learn a model of the environment. Instead, they learn directly from experience by interacting with the environment. Q-learning and Monte Carlo methods are examples of model-free approaches.


## Policy Types

A **policy** ($\pi$) is a strategy that defines how the agent chooses actions based on states.

- **Deterministic policy:** Always selects the same action for a given state.
- **Stochastic policy:** Selects actions according to a probability distribution for each state.

In Q-learning, the optimal policy is typically deterministic, choosing the action with the highest $Q(s, a)$ value for each state. However, during training, stochastic policies (like $\text{epsilon-greedy}$) are often used to encourage exploration.

## Temporal Difference (TD) Learning

Temporal Difference learning is a key method in reinforcement learning that updates value estimates based on the difference between successive predictions. Unlike Monte Carlo methods, TD learning updates values after every step, not just at the end of an episode.

The TD update for the value function is:

$$
v^{\text{new}}(s_t) = v^{\text{old}}(s_t) + \alpha \left[ r_t + \gamma v^{\text{old}}(s_{t+1}) - v^{\text{old}}(s_t) \right]
$$

Where:
- $v^{\text{old}}(s_t)$ is the current estimate of the value of state $s_t$.
- $r_t$ is the reward received after taking an action in $s_t$.
- $s_{t+1}$ is the next state.
- $\gamma$ is the discount factor.
- $\alpha$ is the learning rate.

The TD error $\delta$ is:

$$
\delta = r_t + \gamma v^{\text{old}}(s_{t+1}) - v^{\text{old}}(s_t)
$$

This error measures the difference between the predicted value and the observed reward plus the value of the next state. TD learning uses this error to adjust the value estimate for $s_t$ immediately after each step.

**Advantages of TD Learning:**
- Updates are made online, step-by-step, without waiting for the end of an episode.
- Combines ideas from Monte Carlo and dynamic programming.
- Used in popular algorithms like SARSA and Q-learning.