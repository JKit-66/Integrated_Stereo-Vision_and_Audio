
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

class batchEdit():
    def __init__(self):
        self.directory = 'C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/monday/monday-near-to-far'
        self.ori_file = 'C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/monday/monday-near-to-far/wavfile/20.5cm_0_0.wav'
        self.bg_noise = 'C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/monday/bg_0_0.wav'
        self.editable_file = []
        self.name_prefix = '_'
        self.mono_chann  = 1
        self.p = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paInt16
        self.sample_rate = 44100
        self.indicator = True
        self.ori_spl = 0.0
        
    def pre_spl(self):
        y, _ = librosa.load(self.ori_file,sr=None)
        y_bg, _ = librosa.load(self.bg_noise,sr=None)
        self.ori_spl = self.calculate_spl(y)
        self.noise_spl = self.calculate_spl(y_bg)
        self.plot_noise_db(self.noise_spl)
        theoretical = editor.distance_model(self.ori_spl, 2.10)
        print(theoretical)
        return theoretical
    
    def plot_noise_db(self, spl):
        y_value = spl
        
        plt.axhline(y=y_value, linestyle='--')
        plt.scatter([0], [y_value], color='b') 
        plt.title(f'Amplitude vs Distance (Background Noise)')
        plt.xlabel('Distance (cm)')
        plt.ylabel('Amplitude (dB)')
        plt.xlim([0,210])
        plt.ylim([10,38])
        plt.grid()
        plt.show()


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

    def calculate_spl(self, y):
        # Reference sound pressure (typically 20 µPa in air)
        reference_pressure = 20e-6  #In air, the common reference is 20 μPa, ISBN-10: 0-309-05025-1
        
        # Calculate RMS (Root Mean Square) of the audio signal
        rms = np.sqrt(np.mean(np.square(y)))

        #print(rms)
        # Calculate SPL in decibels
        spl = 20 * np.log10(rms / reference_pressure)
        return spl

    def audio_amplitude_estimate(self, init_dist, init_amp, final_dist):
        # use 10log10() refers to a measure for acoustic energy
        # use 20log10() for amplitude
        p2_p1 = 10**(init_amp/10)
        gap = final_dist/init_dist
        final_p2p1 = (p2_p1/gap**2)
        log_final_p2p1 = 10*np.log10(final_p2p1)
        
        intensity = 1e-12 * 10**(log_final_p2p1/10)
        return log_final_p2p1,intensity
   
    def amplify_editor(self, filenam, factor):
        self.name_prefix = f'amplified_{factor}_'
        song = AudioSegment.from_mp3(filenam)
        louder_song = song + factor
        names = self.name_prefix + os.path.basename(filenam)
        direct = os.path.dirname(filenam)
        louder_song.export(os.path.join(direct, names), format='wav')
        print(os.path.join(direct, names))
        y,_ = librosa.load(os.path.join(direct, names))
        return y
    
    def extract_filenum(self, filename):
        # Split the filename by underscores and periods
        parts = filename.split('_')
        # The number we want is the part before the file extension
        number_str = parts[-1].split('.')[0]
        return float(number_str)
    
    def audio_amplitude_estimate(self, init_dist, init_amp, final_dist):
        # use 10log10() refers to a measure for acoustic energy
        # use 20log10() for amplitude
        p2_p1 = 10**(init_amp/10)
        gap = final_dist/init_dist
        final_p2p1 = (p2_p1/gap**2)
        log_final_p2p1 = 10*np.log10(final_p2p1)
        
        intensity = 1e-12 * 10**(log_final_p2p1/10)
        return log_final_p2p1,intensity

    def distance_model(self, spl, final_d):
        distance = 0.205
        amp_dist = {}
        indB = spl
        while distance <= final_d:
            amp_dist[float(distance)] = indB
            distance += 0.25
            indB, _ = self.audio_amplitude_estimate(0.205,spl,distance)
            #amplitude = indB
        #print(ampli_list, ',' , dist_list)
        return amp_dist


    def get_amplitude_at_distance(self, dist_list, pressure_db, target_distance):
        self.interpolation_function = interpolate.interp1d(dist_list, pressure_db, kind='linear')
        result = self.interpolation_function(target_distance)
        #print('result', result)

        return result
    

    def batch_guestimate_dist(self, dist_list, ampli_list, spl):
        lins = np.linspace(1.3, 1.8, 3)
        result = []
        for i in lins:
            #print('i', i)
            result.append(round(self.get_amplitude_at_distance(dist_list, ampli_list, i) - spl, 3))
        
        return result
   
    def pitch_change_editor(self, data, sampling_rate, pitch_factor):
        y = librosa.effects.pitch_shift(data, sr = sampling_rate, n_steps=pitch_factor)
        self.name_prefix = f'pitch_change_{pitch_factor}_'
        print(type(y))
        return y

    def all_elements_same(self, lst):
        res = lst.count(lst[0]) == len(lst)
        #print(lst, res)
        return lst.count(lst[0]) == len(lst)

    def list_to_graph(self, x,y):

        if len(x) == len(y):
            for i in range(len(x)):
                plt.plot(x[i], y[i], '.-')
            
                plt.title(f'Amplitude vs Distance_({i})')
                plt.xlabel('Distance (cm)')
                plt.ylabel('Amplitude (dB)')
                #plt.legend()
                #plt.xlim([0,225])
                #plt.ylim([15,40])
                plt.grid()
                plt.show()

    def dic_to_graph(self, dic, remark, rem = False):
        x_lab = []
        y_lab = []
        if type(dic) == dict:
            for i in dic:
                x_lab.append(float(i))
                y_lab.append(float(dic[i]))

            if rem == False:
                plt.plot(x_lab, y_lab, '.-')
            else:
                plt.plot([i*100 for i in x_lab], y_lab, '.-')
            
            print(x_lab, y_lab, remark)
            plt.title(f'Amplitude vs Distance ({remark})')
            plt.xlabel('Distance (cm)')
            plt.ylabel('Amplitude (dB)')
            #plt.legend()
            plt.xlim([0,210])
            plt.ylim([10,38])
            plt.grid()
            plt.show()




if __name__ == '__main__':
    editor = batchEdit()
    editor.give_file_direct()
    baseline = editor.pre_spl()
    #editor.editing_feature(editor.editable_file)
    spl_dict = {}
    length_y = []
    fileN = []
    spl_list = []
    res = []
    left_mic = {}  #0 is left, 1 is right
    right_mic = {}  #0 is left, 1 is right

    for j in editor.editable_file:
        #parent_directory = os.path.dirname(j)
        filename = os.path.basename(j)
        print(j)
        numb = editor.extract_filenum(filename)
        fileN.append(filename)
        dist_in_experiment = os.path.splitext(filename)[0].split("cm")[0]
        LoR_in_experiment = os.path.splitext(filename)[0].split("cm")[1]
        dir = '::'
        if editor.indicator == True: #filename == '0_1_15.wav': #editor.editable_file.index(j) == 0: #editor.editable_file.index(j) == 9: #editor.indicator == True: #
            #print(j)
            #editor.resampling(j, 44100)
            y, sr = librosa.load(j,sr=None)
            length_y.append(len(y))
            # Calculate and print SPL
            spl = editor.calculate_spl(y)

            if LoR_in_experiment == '_0_0':
                dir = 'Left' 
            else:
                dir = 'Right' 
            res.append({'File': filename, 'L or R': dir , 'Distance (cm)': str(dist_in_experiment), 'Amplitude (dB)': spl})
            #print('@@@@@@@@@@@spl:::>>', spl, filename)
            spl_dict[numb] = spl           
            spl_list.append(spl)
            #ideal_chg = editor.batch_guestimate_dist(dist, amp, spl)

        if LoR_in_experiment == '_0_0' :
            left_mic[float(dist_in_experiment)] = spl
        elif LoR_in_experiment == '_0_1' :
            right_mic[float(dist_in_experiment)] = spl
        else:
            pass
        

    check = editor.all_elements_same(length_y)
    left_mic = dict(sorted(left_mic.items()))
    right_mic = dict(sorted(right_mic.items()))

    if check == True:
        results_df = pd.DataFrame(res)
        csv_file_path = 'C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/amp_chgwith_dist.csv'
        print(results_df)
        results_df.to_csv(csv_file_path, index=False)
        
        try:
            editor.dic_to_graph(left_mic, 'Left')
            editor.dic_to_graph(right_mic, 'Right')
            theorz = editor.pre_spl()
            editor.dic_to_graph(theorz, 'Theoretical', rem = True)
        except Exception as e:
            print(e)
        #editor.list_to_graph(left_right_dist, left_right_amp)
        #spl_sorted_dict = dict(sorted(spl_dict.items()))
        #see = editor.dic_to_graph(spl_sorted_dict)
    #print(fileN)
