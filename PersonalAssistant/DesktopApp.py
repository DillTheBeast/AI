import pygame
from pygame.locals import *
from Button import Button

# Initialize pygame
pygame.init()

# Set up pygame window
WIDTH, HEIGHT = 800, 600
screen_size = (WIDTH, HEIGHT)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Personal Assistant")

# Set the initial background color to black
background_color = (0, 0, 0)
screen.fill(background_color)

# Define colors
white = (255, 255, 255)
red = (255, 0, 0)

# Draw a white microphone
white_mic_surface = pygame.Surface((100, 50), pygame.SRCALPHA)
pygame.draw.rect(white_mic_surface, (0, 0, 0, 0), (0, 0, 100, 50))  # Set alpha to 0 for full transparency
pygame.draw.circle(white_mic_surface, white, (25, 25), 20, 2)  # Set alpha to 255 for full opacity
pygame.draw.rect(white_mic_surface, white, (50, 15, 15, 20))  # Set alpha to 255 for full opacity

# Draw a red microphone
red_mic_surface = pygame.Surface((100, 50), pygame.SRCALPHA)
pygame.draw.rect(red_mic_surface, (255, 0, 0, 0), (0, 0, 100, 50))  # Set alpha to 0 for full transparency
pygame.draw.circle(red_mic_surface, red, (25, 25), 20)  # Set alpha to 255 for full opacity
pygame.draw.rect(red_mic_surface, red, (50, 15, 15, 20))  # Set alpha to 255 for full opacity

# Render text on top of the background
font = pygame.font.Font(None, 36)
text_surface = font.render("Welcome to Your Personal Assistant", True, (255, 255, 255))  # Set text color to white
text_rect = text_surface.get_rect(center=(WIDTH // 2, 20))
screen.blit(text_surface, text_rect)

# Create a button with the white microphone image
button_rect_white = pygame.Rect((WIDTH // 2 - 50, HEIGHT // 2 - 25, 100, 50))
button_white_mic = Button(button_rect_white, white_mic_surface, (30, 30, 30), (60, 60, 60), lambda: toggle_mic_color([True]))
# Initialize mic color toggle variable
mic_color_white = [True]

def toggle_mic_color(mic_color):
    mic_color[0] = not mic_color[0]
    if mic_color[0]:
        button_white_mic.set_image(white_mic_surface)
        set_background_color((255, 255, 255))  # Set background to white
    else:
        button_white_mic.set_image(red_mic_surface)
        set_background_color((0, 0, 0))  # Set background to black


def set_background_color(color):
    global background_color
    background_color = color
    screen.fill(background_color)

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
            button_white_mic.handle_event(event)

    # Clear the screen
    screen.fill(background_color)

    # Render text on top of the background
    screen.blit(text_surface, text_rect)

    # Draw the button
    button_white_mic.draw(screen, font)

    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()
