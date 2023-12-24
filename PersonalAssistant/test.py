import speech_recognition as sr
from gtts import gTTS
import os
import keyboard
import pygame
import sys

# Define recording_active and inputText as global variables
recording_active = False
inputText = ""

# Initialize the Pygame mixer
pygame.mixer.init()

def TextToSpeech(text):
    # Language in which you want to convert
    language = 'en'

    # Create a gTTS object
    myobj = gTTS(text=text, lang=language, slow=False)

    # Saving the converted audio in an mp3 file named welcome
    myobj.save("audio1.mp3")

    # Load the audio file
    pygame.mixer.music.load("audio1.mp3")

    # Play the audio
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    pygame.time.wait(5000)  # Adjust the wait time as needed

def toggle_recording():
    global recording_active
    recording_active = not recording_active

def SpeechToText():
    # Initialize the recognizer
    r = sr.Recognizer()

    # Use the global keyword to indicate that we want to modify the global variable
    global recording_active
    global inputText

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
                    inputText = r.recognize_google(audio)
                    inputText = inputText.lower()

                    print("Recording:", inputText)

                    # Call TextToSpeech immediately after recognition
                    TextToSpeech(inputText)

                    # Break out of the loop after recognition
                    break

            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

            except sr.UnknownValueError:
                print("Unknown error occurred")

# Call the functions for testing
SpeechToText()