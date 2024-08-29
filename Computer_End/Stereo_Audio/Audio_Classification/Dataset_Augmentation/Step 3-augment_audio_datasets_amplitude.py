
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
#import torch
#from IPython.display import Audio
#from torchaudio.utils import download_asset
#import torchaudio
#import torchaudio.functional as F
import soundfile as sf
#import pyrubberband as pyrb




class batchEdit():
    def __init__(self):
        self.directory = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/2-trimmedaudio'
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
                #print(sub_file)

        #print(self.editable_file) 
        
    def save_file(self, data, save_directory, final_sampling):
        name_prefix = self.name_prefix
        #save_dir = C:/Users/J Kit/Desktop/Y3 Intern/Week3-AudioDataAugmentation/lews\1\guitar_21k.wav
        parent_directory = os.path.dirname(save_directory)  #C:/Users/J Kit/Desktop/Y3 Intern/Week3-AudioDataAugmentation/lews\1
        filename = os.path.basename(save_directory)
        save_filename = name_prefix + filename

        sf.write(os.path.join(parent_directory, save_filename), data, final_sampling)

    
    def calculate_spl(self, y):
        # Reference sound pressure (typically 20 µPa in air)
        reference_pressure = 20e-6  #In air, the common reference is 20 μPa, ISBN-10: 0-309-05025-1
        
        # Calculate RMS (Root Mean Square) of the audio signal
        rms = np.sqrt(np.mean(np.square(y)))

        #print(rms)
        # Calculate SPL in decibels
        spl = 20 * np.log10(rms / reference_pressure)
        return spl

    
    def amplify_editor(self, filenam, factor):
        self.name_prefix = f'amplified_{factor}_'
        song = AudioSegment.from_mp3(filenam)
        louder_song = song + factor
        names = self.name_prefix + os.path.basename(filenam)
        direct = os.path.dirname(filenam)
        louder_song.export(os.path.join(direct, names), format='wav')
        #print(os.path.join(direct, names))
        #y,_ = librosa.load(os.path.join(direct, names))
        return y
    
 


if __name__ == '__main__':
    editor = batchEdit()
    editor.give_file_direct()
    #editor.editing_feature(editor.editable_file)

    for j in editor.editable_file:
        #parent_directory = os.path.dirname(j)
        filename = os.path.basename(j)
        #print(filename)
        #print('@@@@@@@@@@@@@@@@', editor.editable_file)
        #print(os.path.basename(os.path.dirname(j)))
        if (editor.indicator == True): #editor.editable_file.index(j) >= 0:
            #editor.resampling(j, 44100)
            y, sr = librosa.load(j,sr=None)

            # Calculate and print SPL
            spl = editor.calculate_spl(y)
            spl_list = np.linspace(30, spl, 4)
            
            try:
                for i in spl_list:
                    if i < spl:
                        chg = i -spl
                        _ = editor.amplify_editor(j, round(chg,5))
                    else:
                        pass
            except Exception as e:
                print(e)

           

            #y6 = editor.amplify_editor(j, -0.5)
            #y6 = editor.amplify_editor(j, -5)
            #y6 = editor.amplify_editor(j, 0.5)
            #y6 = editor.amplify_editor(j, 5)
            
            #plt.figure(figsize=(10, 4))
            
            #plt.plot(dist,amp, '.-',label='y1 before')
            
            # plt.title(f'Waveform')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')
            #plt.legend()
            #plt.grid()
            #plt.show()
    print('@@@## Step 7 Done !')

