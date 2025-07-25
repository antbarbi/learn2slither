import numpy as np
import random
from enum import Enum


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

DIRECTION = {
    Action.UP:      (-1, 0),
    Action.DOWN:    (1, 0),
    Action.LEFT:    (0, -1),
    Action.RIGHT:   (0, 1)
}

class GameOver(Exception):
    def __init__(self):
        super().__init__("Game Over.")


def _get_random_coordinates() -> tuple[int, int]:
    # Avoid walls (first and last rows/columns)
    return np.random.randint(1, 11), np.random.randint(1, 11)

def _get_random_adjacent(snake: list[tuple[int, int]], coor: tuple[int, int], grid_size: int = 12):
    row, col = coor
    options = [
        (row-1, col), (row+1, col),
        (row, col-1), (row, col+1)
    ]
    valid = [
        (r, c) for r, c in options
        if 1 <= r < grid_size-1 and 1 <= c < grid_size-1 and (r, c) not in snake
    ]
    return random.choice(valid)

class Snake:
    def __init__(self):
        self.reset()

    def _get_dir(self, head_coor: tuple[int, int], segment_coor: tuple[int, int]) -> Action:
        dr = head_coor[0] - segment_coor[0]
        dc = head_coor[1] - segment_coor[1]

        if dr == -1:
            direction = Action.UP
        elif dr == 1:
            direction = Action.DOWN
        elif dc == -1:
            direction = Action.LEFT
        elif dc == 1:
            direction = Action.RIGHT
        return direction

    def _init_snake(self) -> list[tuple[int, int]]:
        coor = _get_random_coordinates()
        self.snake.append(coor)
        for _ in range(2):
            self.snake.append(
                _get_random_adjacent(self.snake, self.snake[-1])
            )
        self.last_action = self._get_dir(self.snake[0], self.snake[1])

    def _random_cell(self) -> tuple[int, int]:
        while True:
            coor = _get_random_coordinates()
            if coor not in self.snake and coor not in self.green_apples and coor != self.red_apple:
                return coor

    def reset(self):
        # Init grid
        self.grid: list[list]                       = np.full((12, 12), '0', dtype='<U1')

        # Fill edges with 'W'
        self.grid[0, :] = 'W'
        self.grid[-1, :] = 'W'
        self.grid[:, 0] = 'W'
        self.grid[:, -1] = 'W'

        self.snake: list[tuple[int, int]]           = []
        self.green_apples: list[tuple[int, int]]    = []
        self.red_apple: tuple[int, int]             = ()

        # Init Snake
        self._init_snake()

        # Init Apples
        for _ in "GG":
            self.green_apples.append(self._random_cell())
        self.red_apple = self._random_cell()

    def step(self, action: Action) -> None:
        dr, dc = DIRECTION[action]
        head_row, head_col = self.snake[0]
        new_row, new_col = head_row + dr, head_col + dc

        # Prevent moving into the second segment
        if len(self.snake) > 1 and (new_row, new_col) == self.snake[1]:
            print("Invalid move: cannot move into the second segment!")
            return

        self.last_action = action
        coor = (new_row, new_col)
        if not (1 <= new_row < self.grid.shape[0] - 1 and 1 <= new_col < self.grid.shape[1] - 1):
            raise GameOver()
        elif coor in self.snake:
            raise GameOver()
        elif coor == self.red_apple:
            if len(self.snake) <= 1:
                raise GameOver()
            self.snake.insert(0, coor)
            self.snake = self.snake[:-2]
            self.red_apple = self._random_cell()
        elif coor in self.green_apples:
            self.snake.insert(0, coor)
            self.green_apples.remove(coor)
            self.green_apples.append(self._random_cell())
        else:
            self.snake.insert(0, coor)
            self.snake.pop()

    def print(self):
        color_map = {
            "W": "\033[33mW\033[0m",    # Yellow
            "H": "\033[34mH\033[0m",    # Blue
            "S": "\033[34mS\033[0m",    # Blue
            "G": "\033[32mG\033[0m",    # Green
            "R": "\033[31mR\033[0m",    # Red
            "0": "0",                   # Default
        }
        grid = self.grid.copy()
        red = self.red_apple
        for index, elem in enumerate(self.snake):
            grid[elem[0], elem[1]] = "H" if index == 0 else "S"
        for elem in self.green_apples:
            grid[elem[0], elem[1]] = "G"
        grid[red[0], red[1]] = "R"
        for row in grid:
            print(" ".join(color_map[cell] for cell in row))
        print()

    def get_observation(self):
        """Returns the visible state in the 4 directions from the snake's head."""
        head_row, head_col = self.snake[0]
        vision = {"up": [], "down": [], "left": [], "right": []}
        # Up
        for r in range(head_row-1, -1, -1):
            vision["up"].append(self.grid[r, head_col])
            if self.grid[r, head_col] == "W":
                break
        # Down
        for r in range(head_row+1, self.grid.shape[0]):
            vision["down"].append(self.grid[r, head_col])
            if self.grid[r, head_col] == "W":
                break
        # Left
        for c in range(head_col-1, -1, -1):
            vision["left"].append(self.grid[head_row, c])
            if self.grid[head_row, c] == "W":
                break
        # Right
        for c in range(head_col+1, self.grid.shape[1]):
            vision["right"].append(self.grid[head_row, c])
            if self.grid[head_row, c] == "W":
                break
        return vision


if __name__ == "__main__":
    snake = Snake()
    INPUT_MAP = {
        "w": Action.UP,
        "s": Action.DOWN,
        "a": Action.LEFT,
        "d": Action.RIGHT
    }
    print(snake.get_observation())
    try:
        while True:
            snake.print()
            snake.step(INPUT_MAP[input()])
    except GameOver as e:
        print(e)