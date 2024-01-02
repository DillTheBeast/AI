import PySimpleGUI as sg
import speech_recognition as sr
import threading
import queue
import pyaudio
from pydub import AudioSegment

# Initialize the recognizer
r = sr.Recognizer()

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

# Initialize the recognizer
r = sr.Recognizer()

def record_audio(q, stop_event):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 0.5  # Adjust the recording duration as needed

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while not stop_event.is_set():
        try:
            data = stream.read(int(RATE / CHUNK * RECORD_SECONDS))
            q.put(data)
        except Exception as e:
            print(f"Error in recording thread: {e}")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    recording_active = False
    q = queue.Queue()
    stop_event = threading.Event()

    layout = [
        [sg.Text("Welcome to your personal Assistant", text_color='white', background_color='gray', justification='center', font=('Helvetica', 16), key='welcome_text', expand_x=True)],
        [sg.Multiline('', size=(50, 4), key='speech_text', font=('Helvetica', 20))],
        [sg.Button("🎤", size=(20, 1.2), font=('Helvetica', 20), key='microphone_button', button_color=('black', 'lightgray')),
         sg.Button("Make both Top", size=(15, 1.2), font=('Helvetica', 20), key='top_button', button_color=('darkgray', 'white'), expand_x=True),
         sg.Button("Make both Bottom", size=(15, 1.2), font=('Helvetica', 20), key='bottom_button', button_color=('white', 'darkgray'), expand_x=True)],
        [sg.Multiline('', key='typed_text', size=(50, 2), font=('Helvetica', 20))],
    ]

    window = sg.Window("Personal Assistant", layout, size=(800, 600), background_color='gray')

    recording_thread = threading.Thread(target=record_audio, args=(q, stop_event), daemon=True)
    recording_thread.start()

    while True:
        event, values = window.read(timeout=500)

        if event == sg.WINDOW_CLOSED:
            stop_event.set()  # Stop the recording thread before closing
            break
        elif event == 'microphone_button':
            recording_active = not recording_active
            window['microphone_button'].update(button_color=('lightgray', 'red') if recording_active else ('red', 'lightgray'))
        elif event == 'top_button':
            window['typed_text'].update(value=values['speech_text'])
        elif event == 'bottom_button':
            window['speech_text'].update(value=values['typed_text'])
        elif event == sg.TIMEOUT_EVENT and recording_active:
            try:
                audio_data = b''.join(q.queue)
                answer = speech_to_text(sr.AudioData(audio_data, 44100, 2))
                if answer:
                    window['speech_text'].update(value=answer[0].upper() + ''.join([answer[i] for i in range(1, len(answer))]) + ". ", append=True)
            except Exception as e:
                print(f"Error in updating text: {e}")

    window.close()

if __name__ == "__main__":
    main()
