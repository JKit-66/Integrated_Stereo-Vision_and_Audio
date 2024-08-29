
#from scipy.io import wavfile
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
import random
import soundfile as sf
#import pyrubberband as pyrb




class batchEdit():
    def __init__(self, direct):
        self.directory = direct
        self.editable_file = []
        self.name_prefix = '_'
        self.mono_chann  = 1
        self.p = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paInt16
        self.sample_rate = 44100
        self.indicator = True
        
    
    def give_file_direct(self):
        direct = self.directory
        main_files = os.listdir(direct)

        for main_file in main_files:
            main_file_dir = os.path.join(direct, main_file)
            sub_files = os.listdir(main_file_dir)

            for sub_file in sub_files:
                self.editable_file.append(os.path.join(main_file_dir, sub_file))

        
    def save_file(self, data, save_directory):
        final_sampling = self.sample_rate
        name_prefix = self.name_prefix
        #save_dir = C:/Users/J Kit/Desktop/Y3 Intern/Week3-AudioDataAugmentation/lews\1\guitar_21k.wav
        parent_directory = os.path.dirname(save_directory)  #C:/Users/J Kit/Desktop/Y3 Intern/Week3-AudioDataAugmentation/lews\1
        filename = os.path.basename(save_directory)
        save_filename = name_prefix + filename

        sf.write(os.path.join(parent_directory, save_filename), data, final_sampling)


    def noise_inject_editor(self, data, noise_factor):
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        augmented_data = augmented_data.astype(type(data[0]))
        self.name_prefix = f'n_{str(noise_factor)}_'
        return augmented_data

    def time_shift_editor(self, data, shift_max, shift_direction):
        sampling_rate = self.sample_rate
        shift = np.random.randint(sampling_rate * shift_max)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(data, shift)
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        
        self.name_prefix = f'ts_{shift_max}_{shift_direction}_'
        return augmented_data
    
   
    def pitch_change_editor(self, data, sampling_rate, pitch_factor):
        y = librosa.effects.pitch_shift(data, sr = sampling_rate, n_steps=pitch_factor)
        self.name_prefix = f'ps_{pitch_factor}_'
        #print(type(y))
        return y

    




if __name__ == '__main__':
    augment = batchEdit('C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-ALong/4-augmentaudio/data')
    augment.give_file_direct()
    noise_opt = [0.0005, 0.005]
    shift_opt = ['right' , 'both']

    for j in augment.editable_file:
        n_option = random.uniform(noise_opt[0], noise_opt[1])
        s_option = random.choice(shift_opt)

        y, sr = librosa.load(j,sr=None)
        y2 = augment.pitch_change_editor(y, 44100, 0.5)
        augment.save_file(y2,j)

        y3A = augment.noise_inject_editor(y, n_option)
        augment.save_file(y3A,j)

        y4 = augment.time_shift_editor(y, 10, s_option)
        augment.save_file(y4,j)

        

