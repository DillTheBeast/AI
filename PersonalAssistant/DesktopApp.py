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
font = pygame.font.Font(None, 32)
userInput = ''

# Set the initial background color to black
background_color = (0, 0, 0)
screen.fill(background_color)

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
inputBox = pygame.Rect(30, 60, 500, 32)
inputBoxActive = (200, 200, 200)
inputBoxPassive = white
currentInputBox = inputBoxPassive

# Draw a white microphone
mic_surface = pygame.Surface((200, 100), pygame.SRCALPHA)
pygame.draw.rect(mic_surface, (0, 0, 0, 0), (50, 5, 20, 40))
pygame.draw.circle(mic_surface, black, (25, 25), 20, 2)
pygame.draw.circle(mic_surface, black, (25, 25), 20)
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
    type_ = black if mic_color_white else white

    pygame.draw.circle(mic_surface, type_, (25, 25), 20)
    pygame.draw.rect(mic_surface, type_, (15, 47, 20, 50))

# Update the display
pygame.display.flip()

# Event loop
running = True
active = False
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if active:
                if event.key == pygame.K_RETURN:
                    # Process the user input
                    print(userInput)
                    userInput = ''
                elif event.key == pygame.K_BACKSPACE:
                    # Remove the last character from the user input
                    userInput = userInput[:-1]
                else:
                    # Append the pressed key to the user input
                    userInput += event.unicode

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if inputBox.collidepoint(event.pos):
                # Button clicked
                print("Button clicked")

    # Calculate the width of the text surface
    text_surface = font.render(userInput, True, black)
    text_width = text_surface.get_width()

    # Update the width and height of the input box
    input_box_width = max(200, text_width + 10)
    input_box_height = 40

    # Update the input box position
    input_box_x = (WIDTH - input_box_width) // 2
    input_box_y = (HEIGHT - input_box_height) // 2

    # Update the button position
    button_x = input_box_x + input_box_width + 20
    button_y = input_box_y
    button_width = 80
    button_height = input_box_height

    # Draw the input box
    pygame.draw.rect(screen, currentInputBox, (input_box_x, input_box_y, input_box_width, input_box_height))

    # Draw the text surface inside the input box
    screen.blit(text_surface, (input_box_x + 5, input_box_y + 5))

    # Draw the button
    pygame.draw.rect(screen, button_color, (button_x, button_y, button_width, button_height))
    button_text = font.render("Submit", True, black)
    screen.blit(button_text, (button_x + 10, button_y + 10))

    pygame.display.update()
    clock.tick(30)

# Quit pygame
pygame.quit()