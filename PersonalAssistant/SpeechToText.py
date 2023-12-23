import speech_recognition as sr
import keyboard

# Initialize the recognizer 
r = sr.Recognizer() 

# Initialize variables
recording = False
text_buffer = []

# Use the microphone as the source for input
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=0.2)

print("Press space to start/stop recording.")

while True:
    try:
        # Listen for the user's input 
        with sr.Microphone() as source2:
            audio2 = r.listen(source2)

        # Use Google to recognize audio
        MyText = r.recognize_google(audio2)
        MyText = MyText.lower()

        if recording:
            text_buffer.append(MyText)
            print("Recording:", MyText)
        
        # Check if spacebar is pressed to toggle recording
        if keyboard.is_pressed("space"):
            recording = not recording
            if recording:
                print("Recording started.")
            else:
                print("Recording stopped. Text:", " ".join(text_buffer))
                text_buffer = []

    except KeyboardInterrupt:
        print("\nExiting the program.")
        break
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    except sr.UnknownValueError:
        print("Unknown error occurred")
