import speech_recognition as sr
from pynput import keyboard as kb
import threading
import time

# Initialize the recognizer
r = sr.Recognizer()

# Initialize variables
recording = False
text_buffer = []

def on_key_release(key):
    global recording, text_buffer
    if key == kb.Key.space:
        recording = not recording
        if recording:
            print("Recording started.")
        else:
            print("Recording stopped. Text:", " ".join(text_buffer))
            text_buffer = []

# Listener for space key
listener = kb.Listener(on_release=on_key_release)
listener.start()

try:
    print("Press space to start/stop recording.")
    while True:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)

        if recording:
            MyText = r.recognize_google(audio).lower()
            text_buffer.append(MyText)
            print("Recording:", MyText)

except KeyboardInterrupt:
    print("\nExiting the program.")
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))
except sr.UnknownValueError:
    print("Speech Recognition could not understand audio")
except Exception as e:
    print("An error occurred: {0}".format(e))

finally:
    listener.stop()
    listener.join()
