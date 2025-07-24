import pygame
from snake import Snake, Action

CELL_SIZE = 64
GRID_SIZE = 10
WINDOW_SIZE = CELL_SIZE * GRID_SIZE

COLOR_BG = (255, 255, 255)
COLOR_SNAKE = (0, 255, 0)      # Green body
COLOR_HEAD = (0, 100, 0)      # Deep green head
COLOR_GREEN_APPLE = (0, 255, 0)
COLOR_RED_APPLE = (255, 0, 0)

def load_images():
    background_img = pygame.image.load("./assets/grass.png")
    background_img = pygame.transform.scale(background_img, (WINDOW_SIZE, WINDOW_SIZE))
    return background_img

def draw_game(screen, snake_engine):
    background_img = load_images()
    screen.blit(background_img, (0, 0))
    # Draw apples
    for apple in snake_engine.green_apples:
        pygame.draw.rect(screen, COLOR_GREEN_APPLE, (apple[1]*CELL_SIZE, apple[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    r = snake_engine.red_apple
    pygame.draw.rect(screen, COLOR_RED_APPLE, (r[1]*CELL_SIZE, r[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    # Draw snake
    for i, segment in enumerate(snake_engine.snake):
        color = COLOR_HEAD if i == 0 else COLOR_SNAKE
        pygame.draw.rect(screen, color, (segment[1]*CELL_SIZE, segment[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
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