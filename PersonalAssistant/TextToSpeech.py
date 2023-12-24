from gtts import gTTS
import os
import pygame

# The text that you want to convert to audio
mytext = input("Input a word or sentence you would like to hear")

# Language in which you want to convert
language = 'en'

# Create a gTTS object
myobj = gTTS(text=mytext, lang=language, slow=False)

# Saving the converted audio in an mp3 file named welcome
myobj.save("audio1.mp3")

# Initialize the Pygame mixer
pygame.mixer.init()

# Load the audio file
pygame.mixer.music.load("audio1.mp3")

# Play the audio
pygame.mixer.music.play()

# Wait for the audio to finish playing
pygame.time.wait(5000)  # Adjust the wait time as needed

# Clean up the Pygame mixer
pygame.mixer.quit()
