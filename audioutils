import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

filename = "output.wav"
fs = 44100   # sample rate (Hz)
channels = 1 # mono

def record(stop_event):
    print("Recording started... Press Enter to stop.")

    recording_list = []

    def callback(indata, frames, time, status):
        if status:
            print("Status:", status)
        recording_list.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
        while not stop_event.is_set():
            sd.sleep(100)

    print("Recording stopped. Saving...")
    recording = np.concatenate(recording_list, axis=0)
    write(filename, fs, recording)
    print(f"Saved as {filename}")
