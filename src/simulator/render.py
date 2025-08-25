import pygame
from simulator.snake import Snake, Action

CELL_SIZE = 64
GRID_SIZE = 12
WINDOW_SIZE = CELL_SIZE * GRID_SIZE

COLOR_BG = (255, 255, 255)
COLOR_SNAKE = (0, 255, 0)      # Green body
COLOR_HEAD = (0, 100, 0)      # Deep green head
COLOR_GREEN_APPLE = (0, 255, 0)
COLOR_RED_APPLE = (255, 0, 0)


def load_images(cell_size):
    images = {
        "head_up":
            pygame.transform.scale(
                pygame.image.load("../assets/head_up.png"),
                (cell_size, cell_size)
                ),
        "head_down":
            pygame.transform.scale(
                pygame.image.load("../assets/head_down.png"),
                (cell_size, cell_size)
                ),
        "head_left":
            pygame.transform.scale(
                pygame.image.load("../assets/head_left.png"),
                (cell_size, cell_size)
                ),
        "head_right":
            pygame.transform.scale(
                pygame.image.load("../assets/head_right.png"),
                (cell_size, cell_size)
                ),
        "tail_up":
            pygame.transform.scale(
                pygame.image.load("../assets/tail_up.png"),
                (cell_size, cell_size)
                ),
        "tail_down":
            pygame.transform.scale(
                pygame.image.load("../assets/tail_down.png"),
                (cell_size, cell_size)
                ),
        "tail_left":
            pygame.transform.scale(
                pygame.image.load("../assets/tail_left.png"),
                (cell_size, cell_size)
                ),
        "tail_right":
            pygame.transform.scale(
                pygame.image.load("../assets/tail_right.png"),
                (cell_size, cell_size)
                ),
        "body_horizontal":
            pygame.transform.scale(
                pygame.image.load("../assets/body_horizontal.png"),
                (cell_size, cell_size)
                ),
        "body_vertical":
            pygame.transform.scale(
                pygame.image.load("../assets/body_vertical.png"),
                (cell_size, cell_size)
                ),
        "body_topleft":
            pygame.transform.scale(
                pygame.image.load("../assets/body_topleft.png"),
                (cell_size, cell_size)
                ),
        "body_topright":
            pygame.transform.scale(
                pygame.image.load("../assets/body_topright.png"),
                (cell_size, cell_size)
                ),
        "body_bottomleft":
            pygame.transform.scale(
                pygame.image.load("../assets/body_bottomleft.png"),
                (cell_size, cell_size)
                ),
        "body_bottomright":
            pygame.transform.scale(
                pygame.image.load("../assets/body_bottomright.png"),
                (cell_size, cell_size)
                ),
        "green_apple":
            pygame.transform.scale(
                pygame.image.load("../assets/green_apple.png"),
                (cell_size, cell_size)
                ),
        "red_apple":
            pygame.transform.scale(
                pygame.image.load("../assets/red_apple.png"),
                (cell_size, cell_size)
                ),
        "wall":
            pygame.transform.scale(
                pygame.image.load("../assets/fire.jpg"),
                (cell_size, cell_size)
                ),
        "background":
            pygame.transform.scale(
                pygame.image.load("../assets/grass.png"),
                (cell_size, cell_size)
                ),
    }
    return images


def get_segment_type(snake: list[tuple[int, int]],
                     i: int, last_action: Action) -> str:
    """Returns the image key for the i-th segment of the snake."""
    if i == 0:  # Head
        if last_action is not None:
            if last_action == Action.UP:
                return "head_up"
            if last_action == Action.DOWN:
                return "head_down"
            if last_action == Action.LEFT:
                return "head_left"
            if last_action == Action.RIGHT:
                return "head_right"
        # fallback to previous logic if last_action is not provided
        if len(snake) > 1:
            dr = snake[0][0] - snake[1][0]
            dc = snake[0][1] - snake[1][1]
            if dr == -1:
                return "head_up"
            if dr == 1:
                return "head_down"
            if dc == -1:
                return "head_left"
            if dc == 1:
                return "head_right"
        return "head_right"
    elif i == len(snake) - 1:  # Tail
        if len(snake) > 1:
            dr = snake[-2][0] - snake[-1][0]
            dc = snake[-2][1] - snake[-1][1]
            if dr == -1:
                return "tail_down"
            if dr == 1:
                return "tail_up"
            if dc == -1:
                return "tail_right"
            if dc == 1:
                return "tail_left"
        return "tail_right"
    else:  # Body
        prev = snake[i - 1]
        curr = snake[i]
        next = snake[i + 1]
        # Check straight
        if prev[0] == curr[0] == next[0]:
            return "body_horizontal"
        if prev[1] == curr[1] == next[1]:
            return "body_vertical"
        # Check corners
        if (prev[0] < curr[0] and next[1] < curr[1]) or (
                next[0] < curr[0] and prev[1] < curr[1]):
            return "body_topleft"
        if (prev[0] < curr[0] and next[1] > curr[1]) or (
                next[0] < curr[0] and prev[1] > curr[1]):
            return "body_topright"
        if (prev[0] > curr[0] and next[1] < curr[1]) or (
                next[0] > curr[0] and prev[1] < curr[1]):
            return "body_bottomleft"
        if (prev[0] > curr[0] and next[1] > curr[1]) or (
                next[0] > curr[0] and prev[1] > curr[1]):
            return "body_bottomright"
        return "body_horizontal"  # fallback


def draw_game(screen, snake_engine):
    images = load_images(CELL_SIZE)

    # Draw the grid: wall on edges, background elsewhere
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if (
                row == 0 or row == GRID_SIZE - 1
                or col == 0 or col == GRID_SIZE - 1
            ):
                screen.blit(images["wall"], (col * CELL_SIZE, row * CELL_SIZE))
            else:
                screen.blit(
                    images["background"],
                    (col * CELL_SIZE,
                     row * CELL_SIZE,
                     CELL_SIZE,
                     CELL_SIZE))

    # Draw apples
    for apple in snake_engine.green_apples:
        screen.blit(
            images["green_apple"],
            (apple[1] * CELL_SIZE,
             apple[0] * CELL_SIZE))
    r = snake_engine.red_apple
    screen.blit(images["red_apple"], (r[1] * CELL_SIZE, r[0] * CELL_SIZE))

    # Draw snake
    for i, segment in enumerate(snake_engine.snake):
        img_key = get_segment_type(
            snake_engine.snake, i, snake_engine.last_action)
        screen.blit(
            images[img_key],
            (segment[1] *
             CELL_SIZE,
             segment[0] *
             CELL_SIZE))

    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Snake Game")
    clock = pygame.time.Clock()
    snake_engine = Snake()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    snake_engine.step(Action.UP)
                elif event.key == pygame.K_s:
                    snake_engine.step(Action.DOWN)
                elif event.key == pygame.K_a:
                    snake_engine.step(Action.LEFT)
                elif event.key == pygame.K_d:
                    snake_engine.step(Action.RIGHT)
        draw_game(screen, snake_engine)
        clock.tick(10)
    pygame.quit()


if __name__ == "__main__":
    main()
