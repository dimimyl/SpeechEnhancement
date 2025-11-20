"""
Simple playback script for enhanced audio.

This file loads 'enhanced_from_librispeech.wav' (created by inference_from_librispeech.py)
using sounddevice and plays it through the default system audio output.

Requirements:
  pip install sounddevice soundfile
"""

import sounddevice as sd
import soundfile as sf

AUDIO_FILE = 'enhanced_from_librispeech.wav'

def play_audio(file_path=AUDIO_FILE):
    try:
        data, samplerate = sf.read(file_path)
        print(f"Playing {file_path} at {samplerate} Hz...")
        sd.play(data, samplerate)
        sd.wait()  # wait until playback is finished
        print("Playback finished.")
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run inference_from_librispeech.py first.")

if __name__ == '__main__':
    play_audio()
