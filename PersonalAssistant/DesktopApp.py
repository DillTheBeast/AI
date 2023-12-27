import pygame
from PIL import Image

# Define Button class
class Button:
    def __init__(self, rect, text, color, highlight_color, callback):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color
        self.highlight_color = highlight_color
        self.callback = callback
        self.highlighted = False
        self.clicked = False

    def draw(self, surface, font):
        # Alternate colors based on the click count
        color = self.highlight_color if self.highlighted else self.color
        if self.clicked == True:
            color = (255 - color[0], 255 - color[1], 255 - color[2])
            textColor = 0
        else:
            textColor = 255
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)
        text_surface = font.render(self.text, True, (textColor, textColor, textColor))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.highlighted = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.clicked = not self.clicked
                self.callback()

# Initialize pygame
pygame.init()

# Set up pygame window
WIDTH, HEIGHT = 800, 600
screen_size = (WIDTH, HEIGHT)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Personal Assistant")

# Set the background color
background_color = (169, 169, 169)
screen.fill(background_color)

# Render text on top of the background
font = pygame.font.Font(None, 36)
text_surface = font.render("Welcome to Your Personal Assistant", True, (0, 0, 0))
text_rect = text_surface.get_rect(center=(WIDTH // 2, 20))
# Blit the text surface onto the screen
screen.blit(text_surface, text_rect)

# Create a button
button_rect = pygame.Rect((WIDTH // 2 - 50, HEIGHT // 2 - 25, 100, 50))
button = Button(button_rect, "Speak", (0, 0, 0), (30, 30, 30), lambda: print("Button clicked"))

# Update the display
pygame.display.flip()

# Event loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        else:
            button.handle_event(event)

    # Clear the screen
    screen.fill(background_color)

    # Render text on top of the background
    screen.blit(text_surface, text_rect)

    # Draw the button
    button.draw(screen, font)

    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()