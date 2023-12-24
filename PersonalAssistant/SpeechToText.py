import speech_recognition as sr
from gtts import gTTS
import os
import keyboard
import pygame
import sys

# Initialize the recognizer
r = sr.Recognizer()

# Flag to indicate whether recording is active
recording_active = False

# Function to toggle recording state
def toggle_recording():
    global recording_active
    recording_active = not recording_active

# Loop infinitely for the user to speak
while True:
    # Check for space key press to toggle recording state
    if keyboard.is_pressed('space'):
        toggle_recording()
        while keyboard.is_pressed('space'):
            pass  # Wait for the space key to be released to avoid multiple toggles

    # Check for escape key press to exit the program
    if keyboard.is_pressed('esc'):
        sys.exit()

    # If recording is active, listen for user's input
    if recording_active:
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=0.2)
                audio = r.listen(source)

                # Using Google to recognize audio
                my_text = r.recognize_google(audio)
                my_text = my_text.lower()

                print("Recording:", my_text)

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("Unknown error occurred")
