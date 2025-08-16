import numpy as np
from typing import Callable, Dict
from .snake import Snake, GameOver, Action, DIRECTION
from collections import deque


def get_heading(env: Snake) -> Action:
    """Return current heading used for relative calculations (prefer last_action then direction)."""
    return getattr(env, "last_action", getattr(env, "direction", Action.UP))

def flatten_state(state_list, grid_size: int = 12):
    """Convert state list (dicts with 'label'/'value') -> normalized numpy vector (float32).
       Accept both:
         - closeness floats in [0,1] (preferred)
         - integer distances (1..N) produced by older code
    """
    out = []
    max_ray = max(2, grid_size - 1)
    for f in state_list:
        label = f.get("label", "")
        val = f.get("value", 0)

        # boolean / danger flags -> 1.0 / 0.0
        if isinstance(val, bool):
            out.append(1.0 if val else 0.0)
            continue

        # distance/closenness features: accept float closeness or integer distance
        if label.startswith("distance_to_"):
            # if caller already provided closeness in [0,1], use it directly
            if isinstance(val, float) or isinstance(val, np.floating):
                v = float(val)
                v = max(0.0, min(1.0, v))
                out.append(v)
                continue
            # otherwise try integer distance -> convert to closeness
            try:
                v_int = int(val)
            except Exception:
                v_int = -1
            if v_int <= 0:
                out.append(0.0)
            else:
                # closeness: adjacent -> 1.0, farthest -> ~1/max_ray
                closeness = (max_ray - (v_int - 1)) / float(max_ray)
                out.append(float(closeness))
            continue

        # fallback numeric cast
        try:
            out.append(float(val))
        except Exception:
            out.append(0.0)
    return np.array(out, dtype=np.float32)

def is_there(arr, elem, grid_size: int = 12) -> float:
    """Return closeness in [0,1] for elem in arr.
       - If elem not found -> 0.0
       - If found at distance d (1 = adjacent) -> closeness = (max_ray - (d-1)) / max_ray
         so adjacent -> 1.0, farthest -> ~1/max_ray
    """
    idx = np.where(arr == elem)[0]
    if idx.size == 0:
        return 0.0
    distance = int(idx[0]) + 1
    max_ray = max(2, grid_size - 1)
    closeness = (max_ray - (distance - 1)) / float(max_ray)
    return float(closeness)

# --- State extraction functions ------------------------------------------------

def distance_state(env: Snake) -> np.ndarray:
        obs = env.get_observation()

        state = [
            {
                "label": "danger_up",
                "value": True if obs["up"][0] == "W" or obs["up"][0] == "S" else False,
            },
            {
                "label": "danger_down",
                "value": True if obs["down"][0] == "W" or obs["down"][0] == "S" else False,
            },
            {
                "label": "danger_left",
                "value": True if obs["left"][0] == "W" or obs["left"][0] == "S" else False,
            },
            {
                "label": "danger_right",
                "value": True if obs["right"][0] == "W" or obs["right"][0] == "S" else False,
            },
            {
                "label": "distance_to_danger_up",
                "value": max(is_there(obs["up"], "W"), is_there(obs["up"], "S")),
            },
            {
                "label": "distance_to_danger_down",
                "value": max(is_there(obs["down"], "W"), is_there(obs["down"], "S")),
            },
            {
                "label": "distance_to_danger_left",
                "value": max(is_there(obs["left"], "W"), is_there(obs["left"], "S")),
            },
            {
                "label": "distance_to_danger_right",
                "value": max(is_there(obs["right"], "W"), is_there(obs["right"], "S")),
            },
            {
                "label": "distance_to_green_up",
                "value": is_there(obs["up"], "G"),
            },
            {
                "label": "distance_to_green_down",
                "value": is_there(obs["down"], "G"),
            },
            {
                "label": "distance_to_green_left",
                "value": is_there(obs["left"], "G"),
            },
            {
                "label": "distance_to_green_right",
                "value": is_there(obs["right"], "G"),
            },
            {
                "label": "distance_to_red_up",
                "value": is_there(obs["up"], "R"),
            },
            {
                "label": "distance_to_red_down",
                "value": is_there(obs["down"], "R"),
            },
            {
                "label": "distance_to_red_left",
                "value": is_there(obs["left"], "R"),
            },
            {
                "label": "distance_to_red_right",
                "value": is_there(obs["right"], "R"),
            },
        ]
        return state

def base_state(env: Snake) -> np.ndarray:
        obs = env.get_observation()

        state = [
            {
                "label": "danger_up",
                "value": True if obs["up"][0] == "W" or obs["up"][0] == "S" else False,
            },
            {
                "label": "danger_down",
                "value": True if obs["down"][0] == "W" or obs["down"][0] == "S" else False,
            },
            {
                "label": "danger_left",
                "value": True if obs["left"][0] == "W" or obs["left"][0] == "S" else False,
            },
            {
                "label": "danger_right",
                "value": True if obs["right"][0] == "W" or obs["right"][0] == "S" else False,
            },
            {
                "label": "distance_to_wall_up",
                "value": is_there(obs["up"], "W"),
            },
            {
                "label": "distance_to_wall_down",
                "value": is_there(obs["down"], "W"),
            },
            {
                "label": "distance_to_wall_left",
                "value": is_there(obs["left"], "W"),
            },
            {
                "label": "distance_to_wall_right",
                "value": is_there(obs["right"], "W"),
            },
            {
                "label": "distance_to_snake_up",
                "value": is_there(obs["up"], "S"),
            },
            {
                "label": "distance_to_snake_down",
                "value": is_there(obs["down"], "S"),
            },
            {
                "label": "distance_to_snake_left",
                "value": is_there(obs["left"], "S"),
            },
            {
                "label": "distance_to_snake_right",
                "value": is_there(obs["right"], "S"),
            },
            {
                "label": "distance_to_green_up",
                "value": is_there(obs["up"], "G"),
            },
            {
                "label": "distance_to_green_down",
                "value": is_there(obs["down"], "G"),
            },
            {
                "label": "distance_to_green_left",
                "value": is_there(obs["left"], "G"),
            },
            {
                "label": "distance_to_green_right",
                "value": is_there(obs["right"], "G"),
            },
            {
                "label": "distance_to_red_up",
                "value": is_there(obs["up"], "R"),
            },
            {
                "label": "distance_to_red_down",
                "value": is_there(obs["down"], "R"),
            },
            {
                "label": "distance_to_red_left",
                "value": is_there(obs["left"], "R"),
            },
            {
                "label": "distance_to_red_right",
                "value": is_there(obs["right"], "R"),
            },
        ]
        return state


# --- Reward functions --------------------------------------------------------
def base_reward(env: Snake, info: dict):
    # tolerate different key names and fall back to env attributes
    info = info or {}
    died = info.get("died", getattr(env, "died", False))
    ate_red = info.get("ate_red", info.get("ate_red_apple", getattr(env, "ate_red", False)))
    ate_green = info.get("ate_green", info.get("ate_green_apple", getattr(env, "ate_green", False)))
    # Terminal death penalty
    if died:
        return -100.0
    # Red apple penalty
    if ate_red:
        return -20.0 * (1.5 * (len(env.snake) + 1))
    # Green apple reward (keep previous scaling)
    if ate_green:
        return 20.0 * (1.5 * len(env.snake))
    # default per-step penalty
    return -0.5

# --- Main class ---------------------------------------------------------------
class SnakeFeatureEngineering:
    STATE_FUNCTIONS = {
        'base': base_state,
        'distance': distance_state,
    }
    
    REWARD_FUNCTIONS = {
        'base': base_reward,
    }
    
    def __init__(self, state_type='base', reward_type='base', history_k=4):
        self.state_fn = self.STATE_FUNCTIONS[state_type]
        self.reward_fn = self.REWARD_FUNCTIONS[reward_type]
        self.total_reward = 0

    def extract_state(self, env):
        return flatten_state(self.state_fn(env))

    def relative_to_action(self, env: Snake, rel_idx: int) -> Action:
        """Map relative index (0=forward,1=left,2=right) -> absolute Action."""
        heading = get_heading(env)
        dr, dc = DIRECTION[heading]
        mapping = {
            0: (dr, dc),        # forward
            1: (-dc, dr),       # left  (90° CCW)
            2: (dc, -dr),       # right (90° CW)
        }
        vec = mapping[int(rel_idx)]
        for act, v in DIRECTION.items():
            if v == vec:
                return act
        return heading

    def step_and_compute_reward(self, env: Snake, action) -> tuple[float, bool]:
        """Perform env.step(action) and return (reward, done).
           Accepts either Action or int for action.
        """
        act = action if isinstance(action, Action) else Action(action)
        try:
            info = env.step(act)
            done = False
        except GameOver as e:
            info = getattr(e, "info", {})
            done = True
        reward = self.reward_fn(env, info)
        self.total_reward += reward
        return float(reward), bool(done)

    def get_relative_action_mask(self, env: Snake):
        """Return boolean mask length 4: True = allowed. Backward action is False."""
        heading = get_heading(env)
        dr, dc = DIRECTION[heading]
        back_vec = (-dr, -dc)
        mask = np.ones(4, dtype=bool)
        for act, v in DIRECTION.items():
            if v == back_vec:
                mask[act.value] = False
                break
        return mask

    def compute_reward(self, env: Snake, info: dict | None = None) -> float:
        """Compute shaped reward using the selected reward function."""
        return float(self.reward_fn(env, info or {}))

# Example usage:
# fe = SnakeFeatureEngineering('extended', 'composite')
# state = fe.extract_state(env)
# reward = fe.compute_reward(env)

if __name__ == "__main__":
    # Interactive loop similar to src/simulator/snake.py main: control with w/s/a/d keys
    print("Interactive feature-engineering demo. Use w/s/a/d to move, q to quit.")
    fe = SnakeFeatureEngineering(state_type='base', reward_type='base')
    env = Snake()
    env.reset()
    INPUT_MAP = {
        "w": (0),  # Action.UP
        "s": (1),  # Action.DOWN
        "a": (2),  # Action.LEFT
        "d": (3),  # Action.RIGHT
    }
    try:
        while True:
            state = fe.extract_state(env)
            print("\nState dim:", state.size, "sample:", state[:12])
            env.print()
            key = input("move (w/a/s/d), q to quit: ").strip()
            if key == 'q':
                break
            if key not in INPUT_MAP:
                print("invalid input")
                continue
            act_idx = INPUT_MAP[key]
            # show relative-action mask and mapping for current heading
            try:
                mask = fe.get_relative_action_mask(env)
            except Exception:
                mask = None
            print("relative-action mask:", mask)
            for rel in (0, 1, 2):
                abs_act = fe.relative_to_action(env, rel)
                print(f" relative {rel} -> absolute {abs_act} vec={DIRECTION[abs_act]}")

            abs_action = Action(act_idx)
            try:
                info = env.step(abs_action)
                done = False
            except GameOver as e:
                info = getattr(e, "info", {})
                done = True
            # print computed reward
            reward = fe.compute_reward(env, info)
            print("reward:", reward, "done:", done, "info:", info)
            if done:
                print("Episode ended, resetting env")
                env.reset()
    except KeyboardInterrupt:
        print("exiting")