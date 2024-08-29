
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
#import torch
#from IPython.display import Audio
#from torchaudio.utils import download_asset
#import torchaudio
#import torchaudio.functional as F
import soundfile as sf
#import pyrubberband as pyrb




class batchEdit():
    def __init__(self):
        self.directory = 'C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/dataset'
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

        #audio_data_int16 = (data * 32767).astype(np.int16)

        #wf =  wave.open(os.path.join(parent_directory, save_filename), 'wb') 
        #wf.setnchannels(self.mono_chann)  # mono
        #wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        #wf.setframerate(self.sample_rate)
        #wf.writeframes(audio_data_int16.tobytes())
        #wf.close()

        sf.write(os.path.join(parent_directory, save_filename), data, final_sampling)

    
    '''def save_for_tensor(self, data, save_directory):
        name_prefix = self.name_prefix
        parent_directory = os.path.dirname(save_directory)  #C:/Users/J Kit/Desktop/Y3 Intern/Week3-AudioDataAugmentation/lews\1
        filename = os.path.basename(save_directory)
        save_filename = name_prefix + filename

        path = os.path.join(parent_directory, save_filename)
        torchaudio.save(path, data, self.sample_rate)
        #inspect_file(path)'''
    
    def calculate_spl(self, y):
        # Reference sound pressure (typically 20 µPa in air)
        reference_pressure = 20e-6  #In air, the common reference is 20 μPa, ISBN-10: 0-309-05025-1
        
        # Calculate RMS (Root Mean Square) of the audio signal
        rms = np.sqrt(np.mean(np.square(y)))

        #print(rms)
        # Calculate SPL in decibels
        spl = 20 * np.log10(rms / reference_pressure)
        return spl
    
    def get_number_of_channels(self, file_path):
        # Open the audio file
        with sf.SoundFile(file_path) as f:
            # Get number of channels
            num_channels = f.channels
        
        return num_channels


    def audio_amplitude_estimate(self, init_dist, init_amp, final_dist):
        # use 10log10() refers to a measure for acoustic energy
        # use 20log10() for amplitude
        p2_p1 = 10**(init_amp/10)
        gap = final_dist/init_dist
        final_p2p1 = (p2_p1/gap**2)
        log_final_p2p1 = 10*np.log10(final_p2p1)
        
        intensity = 1e-12 * 10**(log_final_p2p1/10)
        return log_final_p2p1,intensity

    def noise_inject_editor(self, data, noise_factor):
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        augmented_data = augmented_data.astype(type(data[0]))
        self.name_prefix = f'noise_inject_{str(noise_factor)}_'
        return augmented_data

    def time_shift_editor(self, data, sampling_rate, shift_max, shift_direction):
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
        
        self.name_prefix = f'time_shift_{shift_max}_{shift_direction}_'
        return augmented_data
    
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

    def distance_model(self, spl):
        distance = 0.15
        ampli_list = []
        dist_list = []
        while distance <= 2:
            indB, _ = self.audio_amplitude_estimate(0.15,spl,distance)
            #amplitude = indB
            ampli_list.append(indB)
            dist_list.append(distance)
            distance += 0.25
        #print(ampli_list, ',' , dist_list)
        return ampli_list, dist_list


    def get_amplitude_at_distance(self, dist_list, pressure_db, target_distance):
        self.interpolation_function = interpolate.interp1d(dist_list, pressure_db, kind='linear')
        result = self.interpolation_function(target_distance)
        #print('result', result)

        return result
    
    def check_bit_depth(self, file_path):
        # Open the audio file
        with sf.SoundFile(file_path) as f:
            # Get the subtype (format) of the file
            subtype = f.subtype
            
            # Check if the subtype indicates a 16-bit or 32-bit file
            if 'PCM_16' in subtype:
                return "16-bit"
            elif 'PCM_24' in subtype:
                return "24-bit"
            elif 'PCM_32' in subtype:
                return "32-bit"
            else:
                return subtype

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

    def dic_to_graph(self, dic):
        x_lab = []
        y_lab = []
        if type(dic) == dict:
            for i in dic:
                x_lab.append(float(i))
                y_lab.append(float(dic[i]))

            plt.plot(x_lab, y_lab, '.-')
            
            plt.title('Amplitude vs Distance')
            plt.xlabel('Distance (cm)')
            plt.ylabel('Amplitude (dB)')
            #plt.legend()
            plt.grid()
            plt.show()


    def speed_change_editor(self, data, speed_factor):
        self.name_prefix = f'speed_change_{speed_factor}_'
        return librosa.effects.time_stretch(data, rate = speed_factor)
    

    def resampling(self, file, final_sampling):
        parent_directory = os.path.dirname(file)
        filename = os.path.basename(file)
        original_data, sr = librosa.load(file)
        data2 = librosa.resample(original_data, orig_sr=sr, target_sr=final_sampling)
        local_prefix = f'resample_{final_sampling}_'
        save_filename = local_prefix + filename
        sf.write(os.path.join(parent_directory, save_filename), data2, final_sampling)

    def editing_feature(self, fileN):
        if type(fileN) == list:
            for idx in fileN:
                pass
                #print(f'{idx}, Done!')
            else:
                pass


if __name__ == '__main__':
    editor = batchEdit()
    editor.give_file_direct()
    #editor.editing_feature(editor.editable_file)
    spl_dict = {}
    length_y = []
    fileN = []
    spl_list = []
    results = []
    all_of_classes = []
    unsupported_filetype = ['.m4a']
    for j in editor.editable_file:
        #parent_directory = os.path.dirname(j)
        filename = os.path.basename(j)
        filetype = os.path.splitext(filename)[1]

        if filetype not in unsupported_filetype: #editor.indicator == True: #filename == '0_1_15.wav': #editor.editable_file.index(j) == 0: #editor.editable_file.index(j) == 9: #editor.indicator == True: #
            
            try:
                y, sr = librosa.load(j,sr=None)
                length_y.append(len(y))
                bitt = editor.check_bit_depth(j)
                num_chan = editor.get_number_of_channels(j)
                classname = os.path.basename(os.path.dirname(j))

                if classname not in all_of_classes:
                    all_of_classes.append(classname)

                #print('filename: ', filename,'@@ sr: ', sr, '>> duration: ', round(len(y)/sr,2) , ' ##no of channels: ', num_chan, '$$bytes: ', bitt)
                results.append({'File': filename, 'Class':classname,  'Sampling Rate': sr, 'Duration (s)': round(len(y)/sr,2), 'No of Channels': num_chan,  'Bit Depth': bitt, 'File Type': filetype})
            except AttributeError as e:
                if filetype == '.mp3':
                    audio = AudioSegment.from_file(j, 'mp3')
                    results.append({'File': filename, 'Class':classname,  'Sampling Rate': audio.frame_rate, 'Duration (s)': round(len(audio)/1000,2), 'No of Channels': audio.channels,  'Bit Depth': f'{audio.sample_width * 8}-bit', 'File Type': filetype})
                else:
                    print(e)
                    pass
            else:
                pass
    
    results_df = pd.DataFrame(results)
    print(results_df)
    csv_file_path = 'C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/audio_bit_depths————new-.csv'
    print(all_of_classes)
    results_df.to_csv(csv_file_path, index=False)