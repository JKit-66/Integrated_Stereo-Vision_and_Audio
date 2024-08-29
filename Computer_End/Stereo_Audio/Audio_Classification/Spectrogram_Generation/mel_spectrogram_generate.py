import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random

def generate_mel_spectrogram(wav_dir):
    yes = 'Yes'
    # Process each wav file
    for file in wav_dir:
        print(file)
        idx = random.choice([i for i in range(1,1000)])
        wav_file = file
        file_path = wav_file #os.path.join(wav_dir, wav_file)

        # Load the audio file
        y, sr = librosa.load(file, sr=None)  # Load with original sample rate

        # Generate the Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

        # Convert to log scale (dB). Use the peak power (max) as reference.
        S_dB = librosa.power_to_db(S, ref=np.max)

        print(np.max(y))        
        # Plotting the Mel spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel spectrogram of {wav_file}')
        plt.tight_layout()
        
        # Save the plot as a PNG file
        file_dir = os.path.basename(os.path.dirname(file))
        output_path = f'{file_dir}_spectrogram_{idx}.png'
        plt.savefig(output_path)
        plt.close()

        print(f"Mel spectrogram for {wav_file} saved as {output_path}")

# Example usage
dir_name = 'bellphone'
audio_idx = 28
audio_file_0 = str(audio_idx) + '_0.wav'
audio_file_1 = str(audio_idx) + '_1.wav'

generate_mel_spectrogram([f'./{dir_name}/{audio_file_0}', f'./{dir_name}/{audio_file_1}'])


