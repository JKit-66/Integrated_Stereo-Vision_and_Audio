import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import wave
import pyaudio
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
from scipy import interpolate
import re
import soundfile as sf
import pandas as pd
from pydub import AudioSegment
from scipy.signal import fftconvolve
import pyloudnorm as pyln
#import torch
#from IPython.display import Audio
#from torchaudio.utils import download_asset
#import torchaudio
#import torchaudio.functional as F
import soundfile as sf


def check_TimeDelay(L, R):
    sample_rate = 44100
    fs = sample_rate  # Sampling frequency in Hz
    rateL, dataL = wavfile.read(L)
    rateR, dataR = wavfile.read(R)

    s1 = np.array(dataL)
    s2 = np.array(dataR)

    # Compute the cross-correlation using fftconvolve
    correlation = fftconvolve(s1, s2[::-1], mode='full')

    # Find the lag with the maximum correlation
    lags = np.arange(-len(s1) + 1, len(s1))
    max_lag = lags[np.argmax(correlation)]

    # Convert lag to time delay
    time_delay = max_lag / fs

    return round(time_delay, 6), correlation


fs1, signal1 = wavfile.read('2_0.wav')
fs2, signal2 = wavfile.read('2_1.wav')
times = np.linspace(0,1,fs1)

if fs1 != fs2:
  raise ValueError("Sampling rates of the two signals must be the same.")

time_delay, corr = check_TimeDelay('2_0.wav', '2_1.wav')
print("Estimated time delay:", time_delay, "seconds")

# Plot the cross-correlation function.
plt.plot(corr)
print(len(corr))
plt.xlabel("Time (s)")
plt.ylabel("Audio Signal Amplitude")
plt.title("Plot of Audio Signal Waveform")
plt.legend()
plt.grid()
plt.show()

