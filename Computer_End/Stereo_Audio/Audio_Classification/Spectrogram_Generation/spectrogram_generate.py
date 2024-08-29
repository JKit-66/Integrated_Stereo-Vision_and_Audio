import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_file_path = './drone/6_0.wav'  # Replace with your audio file path
y, sr = librosa.load(audio_file_path, sr=None)

# Compute the Short-Time Fourier Transform (STFT) to generate the spectrogram
D = np.abs(librosa.stft(y))

# Convert amplitude spectrogram to decibel (dB) units for better visualization
S_dB = librosa.amplitude_to_db(D, ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Spectrogram of {audio_file_path}')
plt.show()

# Determine the maximum frequency from the spectrogram
frequencies = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0])

# Find the frequency with the maximum amplitude
max_amplitude_idx = np.argmax(np.mean(D, axis=1))
max_frequency = frequencies[max_amplitude_idx]

print(f"Maximum frequency in the audio file is approximately {max_frequency:.2f} Hz.")
