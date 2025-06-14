import wave
import sys
from pathlib import Path
import pyaudio
import speech_recognition as sr

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 44100
RECORD_SECONDS = 60

def create_wav_data():
    with wave.open('output.wav', 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

        print('Recording...')
        try:
            for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
                wf.writeframes(stream.read(CHUNK))
            print('nRecording timeout')
        except KeyboardInterrupt:
            print("\nRecording canceled")
        

        stream.close()
        p.terminate()

def create_txt_prompt() -> str:
    """
    take in audio data path, create transcription and return textfile path 
    """
    path = Path.cwd() / f"output.wav"
    with sr.AudioFile(str(path)) as source:
        recognizer = sr.Recognizer()
        audio_data = recognizer.record(source)  # Read the audio file
        try:
            text = recognizer.recognize_google(audio_data, language="en-US")
            # text += ". Create a colorful, childlike illustration with a vivid background and clearly defined foreground objects. Simple shapes, playful style, and bold outlines."
            text += ". Create this image with a background clearly separated from the foreground objects. Use Simple shapes and bold outlines."
            text += ". Use additionally only these nine colors: green=(42,167,33), orange=(253,148,73), blue=(77,117,253), red=(236,70,62), yellow=(255,252,35), turquoise=(61,224,228), black=(49,62,63), purple=(223,64,205), light_green=(118,247,109), white=(255,255,255)"
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return ""
        except sr.RequestError:
            print("Error with the recognition service")
            return ""

    return text


if __name__ == '__main__':
    create_wav_data()
    text = create_txt_prompt()
    print(text)