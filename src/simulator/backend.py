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

def _random_cell(grid: np.ndarray, elem: str) -> tuple[int, int]:
    if len(elem) != 1 or elem not in "WHSGR0":
        raise Exception("Elem should be a single character")
    while True:
        row = np.random.randint(0, 10)
        col = np.random.randint(0, 10)
        if grid[row, col] == "0":
            grid[row, col] = elem
            return row, col

def _snake_body(grid: np.ndarray, row: int, col: int) -> tuple[int, int]:
    options = [(row - 1, col), (row, col - 1), (row + 1, col), (row, col + 1)]
    valid_options = [
        (r, c)
        for r, c in options
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == "0"
    ]
    chosen = random.choice(valid_options)
    grid[chosen] = "S"
    return chosen

class Backend:
    def __init__(self):
        self.grid = None
        self.snake = []
        self.reset()
        print(self.grid)
        print(self.snake)

    def reset(self):
        # Init grid
        self.grid = np.full((10, 10), '0', dtype='<U1')
        # Init Snake
        ret = _random_cell(self.grid, "H")
        self.snake.append(ret)
        for _ in "SS":
            ret = _snake_body(self.grid, *ret)
            self.snake.append(ret)
        # Init Apples
        for c in "GGR":
            _random_cell(self.grid, c) # 2 Green and 1 Red

    def _udpate_snake(self, dr, dc, size: int = 0):
        # Change head position
        snake_head = (self.snake[0][0] + dr, self.snake[0][1] + dc)
        self.snake.insert(snake_head)



    def step(self, action: Action) -> None:
        dr, dc = DIRECTION[action]
        head_row, head_col = self.snake[0]
        new_row, new_col = head_row + dr, head_col + dc

        resize = 0
        if not (0 <= new_row < self.grid.shape[0] and 0 <= new_col < self.grid.shape[1]):
            print("game over")
        if self.grid[new_row, new_col] == "S":
            print("game over")
        if self.grid[new_row, new_col] == "R":
            self.snake.pop()
            _random_cell(self.grid, "R")
            resize = -1
        if self.grid[new_row, new_col] == "G":
            resize = 1


    def get_observation():
        pass

    def render():
        pass


if __name__ == "__main__":
    backend = Backend()
    backend.step(Action.DOWN)