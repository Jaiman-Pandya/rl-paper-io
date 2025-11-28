import pygame
from environments.logic import PaperIOGame, Player

class PaperIORenderer:
    def __init__(self, game: PaperIOGame, cell_size: int = 12):
        self.game = game
        self.cell_size = cell_size
        self.width = game.grid_size * cell_size
        self.height = game.grid_size * cell_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Paper.io RL")
        self.clock = pygame.time.Clock()

        self.background = (240, 240, 240)
        self.grid = (220, 220, 220)
        self.trail = (255, 255, 255)

    def render(self):
        self.screen.fill(self.background)

        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.grid, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.grid, (0, y), (self.width, y))

        for player in self.game.players:
            if not player.alive:
                continue

            for x, y in player.territory:
                rectangle = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                darker_r = int(player.color[0] * 0.6)
                darker_g = int(player.color[1] * 0.6)
                darker_b = int(player.color[2] * 0.6)
                color = (darker_r, darker_g, darker_b)

                pygame.draw.rect(self.screen, color, rectangle)

            for x, y in player.trail:
                rectangle = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, player.color, rectangle)

            x_pos = int(player.x * self.cell_size + self.cell_size // 2)
            y_pos = int(player.y * self.cell_size + self.cell_size // 2)

            pygame.draw.circle(self.screen, player.color, (x_pos, y_pos), self.cell_size // 2)

        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.game.players[0].score:.1%}",True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
