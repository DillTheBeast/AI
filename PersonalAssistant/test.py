import PySimpleGUI as sg
import speech_recognition as sr
import audioop
import wave
from pydub import AudioSegment
import threading
import os

recognized_speech = ""

# Initialize the recognizer and the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=0.2)

# Function to toggle recording state
def toggle_recording(window):
    global recording_active
    recording_active = not recording_active

    if recording_active:
        window['microphone_button'].update(button_color=('lightgray', 'red'))
        # Start a new thread for recording
        recording_thread = threading.Thread(target=record_audio, args=(window,), daemon=True)
        recording_thread.start()
    else:
        window['microphone_button'].update(button_color=('red', 'lightgray'))

def speech_to_text(audio):
    try:
        # Using Google to recognize audio
        my_text = r.recognize_google(audio)
        my_text = my_text.lower()
        return my_text
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return None
    except sr.UnknownValueError:
        print("Unknown error occurred")
        return None

def save_to_mp3(audio_data, filename='audio.mp3'):
    wav_filename = 'audio.wav'
    with wave.open(wav_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(audio_data)

    sound = AudioSegment.from_wav(wav_filename)
    sound.export(filename, format="mp3")
    return filename

def record_audio(window):
    global recording_active, recognized_speech
    while recording_active:
        with sr.Microphone() as source:
            audio = r.listen(source)
            audio_data = audioop.tomono(audio.get_raw_data(), audio.sample_width, 0.5, 0.5)  # Convert stereo to mono
            save_to_mp3(audio_data)
            answer = speech_to_text(audio)
            if answer:
                recognized_speech = answer[0].upper() + ''.join([answer[i] for i in range(1, len(answer))]) + ". "


def main():
    global recording_active
    recording_active = False

    layout = [
        [sg.Text("Welcome to your personal Assistant", text_color='white', background_color='gray', justification='center', font=('Helvetica', 16), key='welcome_text', expand_x=True)],
        [sg.Multiline('', size=(50, 4), key='speech_text', font=('Helvetica', 20)), sg.Button("🎤", size=(20, 1.4), font=('Helvetica', 20), key='microphone_button', button_color=('black', 'lightgray'), enable_events=True)],
        [sg.Button("Make both Top", size=(15, 1.2), font=('Helvetica', 20), key='top_button', button_color=('darkgray', 'white'), expand_x=True, enable_events=True), sg.Button("Make both Bottom", size=(15, 1.2), font=('Helvetica', 20), key='bottom_button', button_color=('white', 'darkgray'), expand_x=True, enable_events=True)],
        [sg.Multiline('', key='typed_text', size=(50, 4), font=('Helvetica', 20)), sg.Button("🗣️", size=(20, 1.4), font=('Helvetica', 20), key='speak_button', button_color=('black', 'lightgray'), enable_events=True)]
    ]

    window = sg.Window("Personal Assistant", layout, size=(800, 600), background_color='gray')

    while True:
        event, values = window.read(timeout=500)  # Reduced the update frequency to 500 milliseconds

        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'microphone_button':
            toggle_recording(window)

        elif event == 'update_text':
            # Event received from the recording thread, update the recognized speech in the global variable
            recognized_speech += values[event]
            window['speech_text'].update(value=recognized_speech, append=True)

        elif event == 'speak_button':
            os.system(f'say "{values["typed_text"]}"')

        elif event == 'top_button':
            # Retrieve everything from the bottom + top textbox
            window['typed_text'].update(value=values['speech_text'])

        elif event == 'bottom_button':
            # Retrieve everything from the bottom + top textbox
            window['speech_text'].update(value=values['typed_text'])

    window.close()

if __name__ == "__main__":
    main()
