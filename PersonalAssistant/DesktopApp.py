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
black = (0, 0, 0)
type = black
center = (25, 25)
radius = 20
button_x = 150
button_y = 150
button_size = 100

# Draw a white microphone
mic_surface = pygame.Surface((200, 100), pygame.SRCALPHA)
pygame.draw.rect(mic_surface, (0, 0, 0, 0), (50, 5, 20, 40))
pygame.draw.circle(mic_surface, black, (25, 25), 20, 2)
pygame.draw.circle(mic_surface, black, center, radius)
pygame.draw.rect(mic_surface, black, (15, 47, 20, 50))

# Render text on top of the background
font = pygame.font.Font(None, 36)
text_surface = font.render("Welcome to Your Personal Assistant", True, (255, 255, 255))  # Set text color to white
text_rect = text_surface.get_rect(center=(WIDTH // 2, 20))
screen.blit(text_surface, text_rect)

# Create a button with the white microphone image
button_rect_white = pygame.Rect((WIDTH - 100, 50, 50, 105))
button_white_mic = Button(button_rect_white, mic_surface, (30, 30, 30), (60, 60, 60), lambda: toggle_mic_color())

# Initialize mic color toggle variable
mic_color_white = True

def toggle_mic_color():
    global mic_color_white
    mic_color_white = not mic_color_white
    if mic_color_white:
        type = black
    else:
        type = white
    
    pygame.draw.rect(mic_surface, (0, 0, 0, 0), (50, 5, 20, 40))
    pygame.draw.circle(mic_surface, type, (25, 25), 20, 2)
    pygame.draw.circle(mic_surface, type, center, radius)
    pygame.draw.rect(mic_surface, type, (15, 47, 20, 50))

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

    # Update only the region of the screen where the button is located
    pygame.display.update(button_white_mic.rect)

# Quit pygame
pygame.quit()