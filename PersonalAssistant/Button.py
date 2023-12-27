import pygame

# Define Button class
class Button:
    def __init__(self, rect, image_surface, color, highlight_color, callback):
        self.rect = pygame.Rect(rect)
        self.image_surface = image_surface
        self.color = color
        self.highlight_color = highlight_color
        self.callback = callback
        self.highlighted = False
        self.clicked = True

    def draw(self, surface, font):
        # Alternate colors based on the click count
        color = self.highlight_color if self.highlighted else self.color
        if self.clicked:
            color = (255 - color[0], 255 - color[1], 255 - color[2])

        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2)

        # Draw the image surface
        surface.blit(self.image_surface, self.rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.highlighted = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.clicked = not self.clicked
                self.callback()

    def set_image(self, image_surface):
        self.image_surface = image_surface
