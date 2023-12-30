import PySimpleGUI as sg
import speech_recognition as sr
import audioop
import wave
from pydub import AudioSegment
import threading

# Initialize the recognizer and the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=0.2)

# Function to toggle recording state
def toggle_recording():
    global recording_active
    recording_active = not recording_active

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
    global recording_active
    while recording_active:
        with sr.Microphone() as source:
            audio = r.listen(source)
            audio_data = audioop.tomono(audio.get_raw_data(), audio.sample_width, 0.5, 0.5)  # Convert stereo to mono
            save_to_mp3(audio_data)
            answer = speech_to_text(audio)
            if answer:
                window.write_event_value('update_text', answer[0].upper() + ''.join([answer[i] for i in range(1, len(answer))]) + ". ")

def main():
    global recording_active
    recording_active = False
    recorded_text_list = []

    layout = [
        [sg.Text("Welcome to your personal Assistant", text_color='white', background_color='gray', justification='center', font=('Helvetica', 16), key='welcome_text', expand_x=True)],
        [sg.Multiline('', size=(50, 6), key='input_text', font=('Helvetica', 20)), sg.Button("🎤", size=(20, 1.2), font=('Helvetica', 20), key='microphone_button', button_color=('black', 'lightgray'), enable_events=True)],
    ]

    window = sg.Window("Personal Assistant", layout, size=(800, 600), background_color='gray')

    while True:
        event, values = window.read(timeout=500)  # Reduced the update frequency to 500 milliseconds

        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'microphone_button':
            toggle_recording()
            if recording_active:
                window['microphone_button'].update(button_color=('lightgray', 'red'))
                # Start a new thread for recording
                recording_thread = threading.Thread(target=record_audio, args=(window,), daemon=True)
                recording_thread.start()
            else:
                window['microphone_button'].update(button_color=('red', 'lightgray'))

        elif event == 'update_text':
            # Event received from the recording thread, update the text in the GUI
            window['input_text'].update(value=values[event], append=True)

    window.close()

if __name__ == "__main__":
    main()
