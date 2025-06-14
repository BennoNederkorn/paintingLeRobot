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

def create_txt_prompt() -> Path:
    """
    take in audio data path, create transcription and return textfile path 
    """
    path = Path.cwd() / f"output.wav"
    with sr.AudioFile(str(path)) as source:
        recognizer = sr.Recognizer()
        audio_data = recognizer.record(source)  # Read the audio file
        try:
            text = recognizer.recognize_google(audio_data, language="de-DE")
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