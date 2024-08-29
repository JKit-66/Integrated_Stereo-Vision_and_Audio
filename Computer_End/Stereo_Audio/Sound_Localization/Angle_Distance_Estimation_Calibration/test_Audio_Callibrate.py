
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
#import pyrubberband as pyrb




class batchEdit():
    def __init__(self):
        #self.directory = 'C:/Users/J Kit/Desktop/Y3 Intern/Week7-Audio_Direction_callibrate/audio_Callibrate'#'C:/Users/J Kit/Desktop/Y3 Intern/Week5-AugmentAudiowithReverb/3major_classes'
        self.directory = 'dataset_here' #'@@angle_dist_spl/1sec' # #audio_Callibrate' #test_LorRight'
        #@'C:\Users\J Kit\Desktop\Y3 Intern\Week7-Audio_Direction_callibrate\L_R_020824_test\0_DEG'
        self.editable_file = []
        self.name_prefix = '_'
        self.mono_chann  = 1
        self.p = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paInt16
        self.sample_rate = 44100
        self.indicator = True
    
    def read_wave_file(self, filename):
        with wave.open(filename, 'r') as wav_file:
            # Extract raw audio data
            frames = wav_file.readframes(wav_file.getnframes())
            # Convert raw audio data to numpy array
            amplitude = np.frombuffer(frames, dtype=np.int16)
        return amplitude

    def calculate_average_amplitude(self, amplitude):
        #return np.mean(np.abs(amplitude))
        return np.max(np.abs(amplitude))

    def compare_amplitudes(self, file1, file2):
        # Read the audio files
        amplitude1 = self.read_wave_file(file1)
        amplitude2 = self.read_wave_file(file2)
        
        # Calculate the average amplitude of each file
        avg_amplitude1 = self.calculate_average_amplitude(amplitude1)
        avg_amplitude2 = self.calculate_average_amplitude(amplitude2)* 0.941266
        
        # Compare the average amplitudes
        if avg_amplitude1 > avg_amplitude2:
            return 'L', avg_amplitude1/avg_amplitude2
        else:
            return 'R', avg_amplitude1/avg_amplitude2

    def check_TimeDelay(self, L, R):
        fs = self.sample_rate  # Sampling frequency in Hz
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

        return round(time_delay, 6)
    
    def give_file_direct(self):
        direct = self.directory
        main_files = os.listdir(direct)

        for main_file in main_files:
            main_file_dir = os.path.join(direct, main_file, '1sec')
            sub_files = os.listdir(main_file_dir)

            self.editable_file.append(main_file_dir)
            '''for sub_file in sub_files:
                sub_file_dir = os.path.join(main_file_dir, sub_file)
                self.editable_file.append(sub_file_dir)'''

        #print(self.editable_file)
    
    def calculate_spl(self, j):
        _ , y = wavfile.read(j)
        # Reference sound pressure (typically 20 µPa in air)
        reference_pressure = 20e-6  #In air, the common reference is 20 μPa, ISBN-10: 0-309-05025-1
        
        # Calculate RMS (Root Mean Square) of the audio signal
        if '_0.wav' in os.path.basename(j):
            y = y * 0.941266
        else:
            y = y
        rms = np.sqrt(np.mean(np.square(y)))
        max_ = np.max(y)
        
        #print(rms)
        # Calculate SPL in decibels
        spl = 20 * np.log10(rms / reference_pressure)
        return spl, max_
    
    def extract_filenum(self, filename):
        # Split the filename by underscores and periods
        parts = filename.split('_')
        # The number we want is the part before the file extension
        number_str = parts[-1].split('.')[0]
        return float(number_str)
    
    def get_LUFS(self, path):
        data, rate = sf.read(path)
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data) 
        return loudness

    
    def dict_2_graph(self):
        dB_2_dist = {120.1687895: 440.411,  120.5475001: 936.895, 119.7981034: 1261.334, 120.3636769: 825.47, 120.0640353: 1485.972, 120.2359916: 382.85}
        sorted_dB_2_dist = dict(sorted(dB_2_dist.items(), key=lambda item: item[1]))
        x_ = []
        y_ = []
        for idx in sorted_dB_2_dist:
            x_.append(sorted_dB_2_dist[idx])
            y_.append(idx)
        
        plt.plot(x_, y_)
        plt.xlabel('Distance from Sound Source (mm)')
        plt.ylabel('Sound SPL (dB)')
        plt.grid()
        plt.show()


    def all_elements_same(self, lst):
        res = lst.count(lst[0]) == len(lst)
        #print(lst, res)
        return lst.count(lst[0]) == len(lst)

    def extract_from_csv(self, csv_file, specific_element):
        df = pd.read_csv(csv_file)
        filtered_row = df[df['Idx'] == int(specific_element)]
        print(filtered_row)
        dist_data = filtered_row['Distance (mm)'].values[0]
        ang_data = filtered_row['Angle (o)'].values[0]
        dir_data = filtered_row['Direction'].values[0]

        return dist_data, ang_data, dir_data
        
if __name__ == '__main__':
    editor = batchEdit()
    editor.give_file_direct()
    print('rrr', editor.editable_file)

    #0 is left, 1 is right
    length_y = []
    fileN = []
    results = []
    res2 = []
    louder = 'N'
    louder2 = 'N'
    time_domain = 'P'
    for j in editor.editable_file:
        
        up_up = os.path.basename(os.path.dirname(j))
        
        #print('>>>>>', os.path.basename(j).split('_'))

        #baseName_dir = os.path.basename(j).split('_')[2]
        #baseName_dist = os.path.basename(j).split('_')[1]
        #baseName_ang = os.path.basename(j).split('_')[0]

        baseName_dist, baseName_ang, baseName_dir = editor.extract_from_csv('csv_result/0908_COORD.csv', up_up)
        
        #print('-->', baseName_dir)
        j = j.replace('\\', '/')
        sub_files = os.listdir(j)
        audio_Pair = {}

        for files in sub_files:
            if files.endswith(".wav"):
                parts = files.split('_')
                if len(parts) == 2 and parts[1].endswith(".wav"):
                    key = parts[0]
                    if key in audio_Pair:
                        audio_Pair[key].append(os.path.join(j,files))
                    else:
                        audio_Pair[key] = [os.path.join(j,files)]

        leng = 0
        for pair in audio_Pair:
            leng += 1
            R_audio, L_audio = audio_Pair[pair]
            
            if leng <= 4:
                print(L_audio, R_audio)

            L_filename = os.path.basename(L_audio)
            R_filename = os.path.basename(R_audio)
            L_audio_spl, L_max = editor.calculate_spl(L_audio)
            R_audio_spl, R_max = editor.calculate_spl(R_audio)
            ts_delay  = editor.check_TimeDelay(L_audio, R_audio)

            res2.append({'Directory': L_audio, 'SPL (dB)': editor.calculate_spl(L_audio)})
            res2.append({'Directory': R_audio, 'SPL (dB)': editor.calculate_spl(R_audio)})

            if ts_delay > 0:
                time_domain = 'P'
            else:
                time_domain = 'N'

            if L_audio_spl > R_audio_spl:
                louder = 'L'
            else:
                louder = 'R'
            
            if L_max > R_max:
                louder2 = 'Left'
            else:
                louder2 = 'Right'
            
            ans, ans_r = editor.compare_amplitudes(L_audio, R_audio)
            #results.append({'Directory_L': L_audio, 'Directory_R': R_audio, 'File_L': L_filename, 'File_R': R_filename, 'L_SpL (dB)': L_audio_spl , 'R_SpL (dB)': R_audio_spl, 'Diff in SoL (dB, R-L)': R_audio_spl - L_audio_spl , 'Louder_by_Amp_GPT': editor.compare_amplitudes(L_audio, R_audio) , 'Louder_Dir_SPL': louder, 'Louder_Dir_WVF': louder2, 'Speaker_Dir' : baseName_dir, 'Time_Delay' : ts_delay, 'Time_Delay_Dir' : time_domain})
            results.append({'Index': pair, 'Directory_L': L_audio, 'Directory_R': R_audio, 'File_L': L_filename, 'File_R': R_filename , 
                            'L_SpL (dB)': L_audio_spl , 'R_SpL (dB)': R_audio_spl, 'PercentageDifff_SpL_baseline_R': round((L_audio_spl-R_audio_spl)*100/R_audio_spl,3), 
                            'Avg_SpL (dB)': (L_audio_spl+R_audio_spl)/2, 'Louder?_by_SpL': louder , 'Louder?_by_Amp': ans , 'Max_Amp_Ratio': ans_r , 'Speaker_Direction' : baseName_dir, 'Matches?': baseName_dir.lower() == list(ans)[0].lower(), 
                            'Act_Ang (o)':baseName_ang, 'Act_Dist (mm)':baseName_dist, 
                            'Time_Delay' : ts_delay, 'Time_Delay_Dir' : time_domain})
            #'L_LUFS': editor.get_LUFS(L_audio), 'R_LUFS': editor.get_LUFS(R_audio),

    results_df = pd.DataFrame(results)
    res2_df = pd.DataFrame(res2)
    print(results_df)
    csv_file_path = 'C:/Users/J Kit/Desktop/Y3 Intern/Week7-Audio_Direction_callibrate/csv_result/22-08_audio_bit_L_R_direction.csv'
    csv_file_path2 = 'C:/Users/J Kit/Desktop/Y3 Intern/Week7-Audio_Direction_callibrate/csv_result/22-08_audio_bit_Ind_bits.csv'
    results_df.to_csv(csv_file_path, index=False)
    res2_df.to_csv(csv_file_path2, index=False)

    #editor.dict_2_graph()

    
