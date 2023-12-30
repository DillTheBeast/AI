import PySimpleGUI as sg

def main():
    recording = False
    layout = [
        [sg.Text("Welcome to your personal Assistant", text_color='white', background_color='gray', justification='center', font=('Helvetica', 16), key='welcome_text', expand_x=True)],
        [sg.Multiline('', size=(50, 6), key='input_text', pad=((0, 0), (10, 0)), font=('Helvetica', 20)), sg.Button("🎤", size=(4, 1.05), font=('Helvetica', 40), key='microphone_button', pad=((5, 0), (10, 0)), button_color=('black', 'lightgray'), enable_events=True)],
    ]

    window = sg.Window("Personal Assistant", layout, size=(800, 600), background_color='gray')

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'microphone_button':
            recording = not recording
            if recording:
                window['microphone_button'].update(button_color=('lightgray', 'red'))
            else:
                window['microphone_button'].update(button_color=('red', 'lightgray'))

    window.close()

if __name__ == "__main__":
    main()
