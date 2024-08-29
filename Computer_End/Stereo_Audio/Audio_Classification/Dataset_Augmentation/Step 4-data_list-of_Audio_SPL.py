import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
#import wave
import pyaudio
#from pydub import AudioSegment
#from scipy.io import wavfile
#from scipy import signal
import time
import pandas as pd
from collections import Counter

class batchAnalysis():
    def __init__(self):
        self.directory = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/2-trimmedaudio' #'3major_classes'
        self.editable_file = []
        self.mono_chann  = 1
        self.p = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paInt16
        self.sample_rate = 44100
    
    def give_file_direct(self):     
        direct = self.directory
        main_files = os.listdir(direct)

        for main_file in main_files:
            main_file_dir = os.path.join(direct, main_file)
            sub_files = os.listdir(main_file_dir)

            for sub_file in sub_files:
                self.editable_file.append(os.path.join(main_file_dir, sub_file))
    
    # Define a function to check if an audio clip is quiet
    def is_quiet(self, audio_file): #, threshold_db=-40.0):
        y, _ = librosa.load(audio_file, sr=None)
        # Compute the mean of the absolute value of the audio signal
        rms = np.sqrt(np.mean(y**2))
        # Convert to decibels
        db = librosa.amplitude_to_db([rms])[0]
        return db

    def calculate_spl(self, audio_file):
        y, _ = librosa.load(audio_file, sr=None)
        # Reference sound pressure (typically 20 µPa in air)
        reference_pressure = 20e-6  #In air, the common reference is 20 μPa, ISBN-10: 0-309-05025-1
        
        # Calculate RMS (Root Mean Square) of the audio signal
        rms = np.sqrt(np.mean(np.square(y)))

        #print(rms)
        # Calculate SPL in decibels
        spl = 20 * np.log10(rms / reference_pressure)
        return spl
        


if __name__ == '__main__':
    editor = batchAnalysis()
    editor.give_file_direct()
    stat_list = []
    results = []

    start_time = time.time()

    #FIL = 'C:/Users/J Kit/Desktop/Y3 Intern/Week5-PrepareAudioDataset_AugmentAudiowithReverb/backup_backup_3major_classes/backup/3major_classes/telefon/amplified_-12.97392_Chain_0_0_-_Chain_0_1_-_Chain_BckgNoise_apply_-_segment_454481__helloima7__doctors-office-phone.wav_7.wav'
    #y, _ = librosa.load(FIL, sr=None)
    #print(y)

    for j in editor.editable_file:
        filename = os.path.basename(j)
        classname = os.path.basename(os.path.dirname(j))
        spl_now = editor.is_quiet(j)
        spl_too = editor.calculate_spl(j)
        results.append({'Directory': j, 'File': filename, 'Class':classname, 'Sound Pressure Level (dB)':spl_now, 'Sound Pressure Level w Reference Pressure (dB)':spl_too})
    
    results_df = pd.DataFrame(results)
    print(results_df)
    
    csv_file_path = 'Trimmed_audio_spl_details.csv'
    results_df.to_csv(os.path.join(os.path.dirname(editor.directory), 'csv_result', csv_file_path), index=False)

    end_time = time.time()

    print(f"Total usage time: {end_time-start_time} seconds")
    print('@@@## Step 8 Done !')


    df = pd.read_csv(os.path.join(os.path.dirname(editor.directory), 'csv_result', csv_file_path)) #[]
    filtered_df = df[(df['Sound Pressure Level w Reference Pressure (dB)'] < 30.017) | (df['Class'] == 'telefon')]
    rows, columns = filtered_df.shape
    directory_list = filtered_df['Directory'].tolist()
    if rows == len(directory_list):
        ans = input(f'Confirm to Delete {len(directory_list)} files [Y/N] :')
        if ans == 'Y' or ans == 'y':
            for direct in directory_list:
                os.remove(direct)
            print('@@@## Step 8 Done !')
        else:
            print('@@@## Aborted: Step 8 UnDone!')
    else:
        print('ERROR! Not having same amount!?')
