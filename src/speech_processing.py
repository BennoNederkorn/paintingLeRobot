
import numpy as np
import speech_recognition as sr
from typing import Optional



def execute_recording() -> Optional[list[float]]:
    """
    Records audio while a key is pressed and returns the audio data
    Returns:
        Optional[list[float]]: Raw audio data or None if recording failed
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Press Ctrl+C to stop recording")
        try:
            audio_data = recognizer.listen(source)
            return audio_data.get_raw_data()
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None


def speech_to_text(audio_data: Optional[list[float]]) -> str:
    """
    Converts speech data to text using Google's speech recognition
    Args:
        audio_data: Raw audio data from recording
    Returns:
        str: Transcribed text, or error message if transcription fails
    """
    if audio_data is None:
        return "Error: No audio data provided"
    
    recognizer = sr.Recognizer()
    try:
        # Convert raw audio data to AudioData object
        audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
        # Use Google's speech recognition
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with speech recognition service; {e}"