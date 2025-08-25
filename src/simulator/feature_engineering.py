import numpy as np
from .snake import Snake, GameOver, Action, DIRECTION
from collections import deque


def get_heading(env: Snake) -> Action:
    """Return current heading used for relative calculations
    (prefer last_action then direction)."""
    return getattr(env, "last_action", getattr(env, "direction", Action.UP))


def flatten_state(state_list, grid_size: int = 12):
    """Convert state list (dicts with 'label'/'value')
        -> normalized numpy vector (float32).
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

        # distance/closenness features: accept float closeness or integer
        # distance
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
       - If found at distance d (1 = adjacent)
        -> closeness = (max_ray - (d-1)) / max_ray
         so adjacent -> 1.0, farthest -> ~1/max_ray
    """
    idx = np.where(arr == elem)[0]
    if idx.size == 0:
        return 0.0
    distance = int(idx[0]) + 1
    max_ray = max(2, grid_size - 1)
    closeness = (max_ray - (distance - 1)) / float(max_ray)
    return float(closeness)

# --- State extraction functions -----------------------------------------


def relative_state(env: Snake) -> np.ndarray:
    obs = env.get_observation()

    # Map absolute observations to relative directions based on heading
    heading = get_heading(env)
    if heading == Action.UP:
        rel = {
            'forward': obs['up'],
            'back': obs['down'],
            'left': obs['left'],
            'right': obs['right']}
    elif heading == Action.DOWN:
        rel = {
            'forward': obs['down'],
            'back': obs['up'],
            'left': obs['right'],
            'right': obs['left']}
    elif heading == Action.LEFT:
        rel = {
            'forward': obs['left'],
            'back': obs['right'],
            'left': obs['down'],
            'right': obs['up']}
    else:
        rel = {
            'forward': obs['right'],
            'back': obs['left'],
            'left': obs['up'],
            'right': obs['down']}

    state = [
        {
            "label": "danger_forward",
            "value":
                True if rel["forward"][0] == "W" or rel["forward"][0] == "S"
                else False
        },
        {
            "label": "danger_back",
            "value":
                True if rel["back"][0] == "W" or rel["back"][0] == "S"
                else False
        },
        {
            "label": "danger_left",
            "value":
                True if rel["left"][0] == "W" or rel["left"][0] == "S"
                else False
        },
        {
            "label": "danger_right",
            "value":
                True if rel["right"][0] == "W" or rel["right"][0] == "S"
                else False
        },
        {
            "label": "distance_to_wall_forward",
            "value": is_there(rel["forward"], "W")
        },
        {
            "label": "distance_to_wall_back",
            "value": is_there(rel["back"], "W")
        },
        {
            "label": "distance_to_wall_left",
            "value": is_there(rel["left"], "W")
        },
        {
            "label": "distance_to_wall_right",
            "value": is_there(rel["right"], "W")
        },

        {
            "label": "distance_to_snake_forward",
            "value": is_there(rel["forward"], "S")
        },
        {
            "label": "distance_to_snake_back",
            "value": is_there(rel["back"], "S")
        },
        {
            "label": "distance_to_snake_left",
            "value": is_there(rel["left"], "S")
        },
        {
            "label": "distance_to_snake_right",
            "value": is_there(rel["right"], "S")
        },

        {
            "label": "distance_to_green_forward",
            "value": is_there(rel["forward"], "G")
        },
        {
            "label": "distance_to_green_back",
            "value": is_there(rel["back"], "G")
        },
        {
            "label": "distance_to_green_left",
            "value": is_there(rel["left"], "G")
        },
        {
            "label": "distance_to_green_right",
            "value": is_there(rel["right"], "G")
        },
        {
            "label": "distance_to_red_forward",
            "value": is_there(rel["forward"], "R")
        },
        {
            "label": "distance_to_red_back",
            "value": is_there(rel["back"], "R")
        },
        {
            "label": "distance_to_red_left",
            "value": is_there(rel["left"], "R")
        },
        {
            "label": "distance_to_red_right",
            "value": is_there(rel["right"], "R")
        },
    ]
    return state


def base_state(env: Snake) -> np.ndarray:
    obs = env.get_observation()

    state = [
        {
            "label": "danger_up",
            "value":
                True if obs["up"][0] == "W" or obs["up"][0] == "S"
                else False,
        },
        {
            "label": "danger_down",
            "value":
                True if obs["down"][0] == "W" or obs["down"][0] == "S"
                else False,
        },
        {
            "label": "danger_left",
            "value":
                True if obs["left"][0] == "W" or obs["left"][0] == "S"
                else False,
        },
        {
            "label": "danger_right",
            "value":
                True if obs["right"][0] == "W" or obs["right"][0] == "S"
                else False,
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
    ate_red = info.get(
        "ate_red", info.get(
            "ate_red_apple", getattr(
                env, "ate_red", False)))
    ate_green = info.get(
        "ate_green", info.get(
            "ate_green_apple", getattr(
                env, "ate_green", False)))
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

# --- Main class ---------------------------------------------------------


class SnakeFeatureEngineering:
    STATE_FUNCTIONS = {
        'base': base_state,
        'relative': relative_state,
    }

    REWARD_FUNCTIONS = {
        'base': base_reward,
    }

    def __init__(self, state_type='base', reward_type='base', history_k=5):
        self.state_fn = self.STATE_FUNCTIONS[state_type]
        self.reward_fn = self.REWARD_FUNCTIONS[reward_type]
        self.total_reward = 0
        self.history_k = history_k
        self._hist = deque(maxlen=history_k)

    def reset_history(self, env):
        # start with a blank (zero) history so the earliest frames contain no
        # info
        s = flatten_state(self.state_fn(env))
        zero = np.zeros_like(s, dtype=np.float32)
        self._hist.clear()
        for _ in range(self.history_k):
            self._hist.append(zero.copy())

    def extract_state(self, env):
        s = flatten_state(self.state_fn(env))
        self._hist.append(s)
        return np.concatenate(list(self._hist), axis=0)

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

    def step_and_compute_reward(
            self, env: Snake, action) -> tuple[float, bool]:
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
        """Return boolean mask length 4: True = allowed.
        Backward action is False."""
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
    # Interactive feature-engineering demo.
    # Accepts absolute (w/a/s/d) and relative (0/1/2)
    # moves. Prints FE flattened state each step.
    fe = SnakeFeatureEngineering(
        state_type='base',
        reward_type='base',
        history_k=1)
    env = Snake()
    env.reset()
    fe.reset_history(env)

    INPUT_MAP = {
        "w": Action.UP,
        "s": Action.DOWN,
        "a": Action.LEFT,
        "d": Action.RIGHT,
    }

    def _relative_to_action(last_action: Action, rel_idx: int) -> Action:
        heading = last_action if isinstance(last_action, Action) else Action.UP
        dr, dc = DIRECTION[heading]
        mapping = {
            0: (dr, dc),
            1: (-dc, dr),
            2: (dc, -dr),
        }
        vec = mapping.get(int(rel_idx), (dr, dc))
        for act, v in DIRECTION.items():
            if v == vec:
                return act
        return heading

    print(
        "Interactive feature-engineering demo",
        "Use w/s/a/d for absolute moves, 0/1/2 for relative moves, q to quit."
    )
    try:
        while True:
            state = fe.extract_state(env)
            debug_state = relative_state(env)
            print("\nFE state dim:", state.size)
            env.print()
            obs = env.get_observation()
            print("Observation:")
            for k, v in obs.items():
                print(f"  {k}: {''.join(list(v))}")
            print(f"Snake coords: {env.snake}")
            print(
                f"Last action: {getattr(env, 'last_action', None)}"
                f"| Last event: {getattr(env, 'last_event', None)}"
            )
            from pprint import pprint
            pprint(debug_state)

            raw = input("Action> ").strip().lower()
            if raw == "q":
                print("Quitting interactive session.")
                break

            if raw in INPUT_MAP:
                action = INPUT_MAP[raw]
            elif raw in ("0", "1", "2"):
                action = _relative_to_action(
                    getattr(env, 'last_action', Action.UP), int(raw))
            else:
                print("Unknown input. Use w/s/a/d or 0/1/2. q to quit.")
                continue

            try:
                info = env.step(action)
                done = False
            except GameOver as e:
                info = getattr(e, "info", {})
                done = True

            reward = fe.compute_reward(env, info)
            print(f"reward: {reward}, done: {done}, info: {info}")
            if done:
                print("Episode ended, resetting env")
                env.reset()
                fe.reset_history(env)
    except KeyboardInterrupt:
        print("exiting")
