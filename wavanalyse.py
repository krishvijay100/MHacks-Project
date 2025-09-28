import numpy as np
from scipy.io import wavfile

def analyze_wav(filename="output.wav"):
    sr, y = wavfile.read(filename) # sr = sample rate, y = numpy array

    if y.ndim > 1:  # convert to mono if stereo
        y = y.mean(axis=1)

    y = y.astype(np.float32) # make the range from [-1.1]
    y = y / np.max(np.abs(y))

    pauses = 0
    threshold = 0.2
    min_pause_len = int(0.1 * sr) # gives it a 0.1s tolerance

    is_silent = np.abs(y) < threshold
    count = 0

    for val in is_silent:
        if val:
            count += 1
        else:
            if count >= min_pause_len:
                pauses += 1
            count = 0

    duration = len(y) / sr
    return pauses, duration
