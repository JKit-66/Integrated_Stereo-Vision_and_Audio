
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
    def __init__(self, fileNames):
        self.directory = fileNames
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


    def save_file2(self, data, save_directory, final_sampling, name):
        sf.write(os.path.join(save_directory, name), data, final_sampling)

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

    def split_mp3_to_wav(self, mp3_file_path):
        # Load the MP3 file

        classname = os.path.basename(os.path.dirname(mp3_file_path))
        parent_directory = "C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/2-trimmedaudio"
        parent_directory = os.path.join(parent_directory, classname)

        audio = AudioSegment.from_mp3(mp3_file_path)
        
        # Get the duration of the audio in milliseconds
        duration_ms = len(audio)
        
        # Split the audio into one-second segments
        for i in range(0, duration_ms, 1000):
            segment = audio[i:i+1000]
            segment = segment.set_frame_rate(self.sample_rate)
            output_file = f'segment_{filename}_{i+1}.wav'
            output_path = os.path.join(parent_directory, output_file)
            segment.export(output_path, format="wav")


    def trim_file(self, in_file):
        input_file = in_file
        classname = os.path.basename(os.path.dirname(in_file))
        parent_directory = "C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/2-trimmedaudio"
        parent_directory = os.path.join(parent_directory, classname)
        sr, data = wavfile.read(input_file)

        # Determine the length of the audio in seconds
        total_length_seconds = len(data) / sr

        # Calculate the number of one-second segments
        num_segments = int(total_length_seconds)

        # Loop through the audio and save each one-second segment
        for i in range(num_segments):
            start_sample = i * sr
            end_sample = (i + 1) * sr
            segment_data = data[start_sample:end_sample]
            # Define the output file name
            filename = os.path.basename(in_file)
            output_file = f'segment_{filename}_{i+1}.wav'
            # Save the one-second segment
            self.save_file2(segment_data, parent_directory, self.sample_rate, output_file)

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



class identifyImposter():
    def __init__(self, mainfileN):
        #self.main_fileN = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/csv_result/audio_bit_depths—Latest.csv'
        mainfileN2 = mainfileN.replace("\\", '/')
        self.df = pd.read_csv(mainfileN2)
        self.sampling_rate = 44100
        self.duration = 1.0

    def identify_diff_duration(self):
        filtered_df = self.df[self.df['Duration (s)'] != self.duration]
        filenames = filtered_df['Directory'].tolist()
        return filenames

    def identify_diff_sr(self):
        filtered_df = self.df[self.df['Sampling Rate'] != self.sampling_rate]
        filenames = filtered_df['Directory'].tolist()
        return filenames
    
    def re_duration(self, file):
        
        file = file.replace('\\', '/')
        audio = AudioSegment.from_file(file)
        audio_added = AudioSegment.from_file("C:/Users/J Kit/Desktop/Y3 Intern/Week4-DistanceModelPhysical/monday/OverlayNoises/0_1.wav")
        duration_ms = len(audio)
        if duration_ms < 1000:
            padding_needed = 1000 - duration_ms
            audio = audio + audio_added[:padding_needed]
        elif duration_ms > 1000:
            audio = audio[:1000]

        parent_directory = os.path.dirname(file)
        filename = os.path.basename(file)
        local_prefix = f'reduration_1sec_'
        save_filename = local_prefix + filename
        audio.export(os.path.join(parent_directory, save_filename), format='wav')
        os.remove(file)
    
    

    def resampling(self, file, final_sampling):
        parent_directory = os.path.dirname(file)
        filename = os.path.basename(file)
        original_data, sr = librosa.load(file)
        data2 = librosa.resample(original_data, orig_sr=sr, target_sr=final_sampling)
        local_prefix = f'resample_{final_sampling}_'
        save_filename = local_prefix + filename
        sf.write(os.path.join(parent_directory, save_filename), data2, final_sampling)
        os.remove(file)



if __name__ == '__main__':
    most_main = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/downloaded_dataset'
    trim_audio_newpath = "C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/2-trimmedaudio"
    editor = batchEdit(most_main)
    editor.give_file_direct()
    spl_dict = {}
    length_y = []
    fileN = []
    spl_list = []
    all_of_classes = []
    results = []
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
    csv_filename = 'Main_audio_bit_depths—Latest.csv'
    csv_file_path = os.path.join(os.path.dirname(editor.directory), 'csv_result', csv_filename)
    results_df.to_csv(csv_file_path, index=False)
    print(csv_file_path)

    print(all_of_classes)
    print('@@@## Step 1 Done !')

    for classs in all_of_classes:
        parent_trim_directory = os.path.join(trim_audio_newpath, classs)

        if not os.path.exists(parent_trim_directory):
            os.makedirs(parent_trim_directory)
            print(f'Created Folders {parent_trim_directory}')

    print('@@@## Step 2 Done !')

    for j in editor.editable_file:
        filename = os.path.basename(j)
        filetype = os.path.splitext(filename)[1]
        classname = os.path.basename(os.path.dirname(j))

        if filetype not in unsupported_filetype and ( classname == 'jogging'):
            try:
                editor.trim_file(j)
            except Exception as e:
                editor.split_mp3_to_wav(j)
        
    print('@@@## Step 3 Done !')

    most_maintoo = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/2-trimmedaudio'
    editortoo = batchEdit(most_maintoo)
    editortoo.give_file_direct()
    
    length_y = []
    fileN = []
    results = []
    for j in editortoo.editable_file:
        #parent_directory = os.path.dirname(j)
        filename = os.path.basename(j)
        filetype = os.path.splitext(filename)[1]

        if editortoo.indicator == True:
            #print(j)
            try:
                y, sr = librosa.load(j,sr=None)
                length_y.append(len(y))
                bitt = editortoo.check_bit_depth(j)
                num_chan = editortoo.get_number_of_channels(j)
                classname = os.path.basename(os.path.dirname(j))
                results.append({'Directory': j, 'File': filename, 'Class':classname,  'Sampling Rate': sr, 'Duration (s)': round(len(y)/sr,2), 'No of Channels': num_chan,  'Bit Depth': bitt, 'File Type': filetype})
            except AttributeError as e:
                if filetype == '.mp3':
                    audio = AudioSegment.from_file(j, 'mp3')
                    results.append({'Directory': j, 'File': filename, 'Class':classname,  'Sampling Rate': audio.frame_rate, 'Duration (s)': round(len(audio)/1000,2), 'No of Channels': audio.channels,  'Bit Depth': f'{audio.sample_width * 8}-bit', 'File Type': filetype})
                else:
                    pass
            else:
                pass
    
    results_df = pd.DataFrame(results)
    print(results_df)

    csv_filename = 'Trimmed_audio_bit_depths—Before.csv'
    csv_file_path = os.path.join(os.path.dirname(editortoo.directory), 'csv_result', csv_filename)
    results_df.to_csv(csv_file_path, index=False)

    print('@@@## Step 4 Done !')

    identifier = identifyImposter(csv_file_path) 
    identifier.main_fileN = csv_file_path
    files_sr = identifier.identify_diff_sr()
    files_dur = identifier.identify_diff_duration()

    ans = input(f'Modify {len(files_sr)} files of different sampling rate and {len(files_dur)} of different duration [Y/N] ?: ') 
    if (ans == 'Y') or (ans == 'y'):
        print('Yes')
        for file in files_sr:
            identifier.resampling(file, identifier.sampling_rate)
        for file in files_dur:
            identifier.re_duration(file)
        print('!!!!!!!!! Success-resampled and re-durationed')
    else:
        pass

    most_maintoo = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioAugmentation-一条龙/2-trimmedaudio'
    editortoo = batchEdit(most_maintoo)
    editortoo.give_file_direct()
    
    length_y = []
    fileN = []
    results = []
    for j in editortoo.editable_file:
        #parent_directory = os.path.dirname(j)
        filename = os.path.basename(j)
        filetype = os.path.splitext(filename)[1]

        if editortoo.indicator == True:
            #print(j)
            try:
                y, sr = librosa.load(j,sr=None)
                length_y.append(len(y))
                bitt = editortoo.check_bit_depth(j)
                num_chan = editortoo.get_number_of_channels(j)
                classname = os.path.basename(os.path.dirname(j))
                results.append({'Directory': j, 'File': filename, 'Class':classname,  'Sampling Rate': sr, 'Duration (s)': round(len(y)/sr,2), 'No of Channels': num_chan,  'Bit Depth': bitt, 'File Type': filetype})
            except AttributeError as e:
                if filetype == '.mp3':
                    audio = AudioSegment.from_file(j, 'mp3')
                    results.append({'Directory': j, 'File': filename, 'Class':classname,  'Sampling Rate': audio.frame_rate, 'Duration (s)': round(len(audio)/1000,2), 'No of Channels': audio.channels,  'Bit Depth': f'{audio.sample_width * 8}-bit', 'File Type': filetype})
                else:
                    pass
            else:
                pass
    
    results_df = pd.DataFrame(results)
    print(results_df)

    all_dur = []
    all_sr = []
    csv_filename = 'Trimmed_audio_bit_depths—After.csv'
    csv_file_path2 = os.path.join(os.path.dirname(editortoo.directory), 'csv_result', csv_filename)
    results_df.to_csv(csv_file_path2, index=False)
    
    csv_file_path2 = csv_file_path2.replace("\\", '/')
    df3 = pd.read_csv(csv_file_path2)
    filtered_df_dur = df3['Duration (s)'].tolist()
    filtered_df_sr = df3['Sampling Rate'].tolist()

    if (len(set(filtered_df_dur)) == len(set(filtered_df_sr))) and (len(set(filtered_df_sr)) == 1):
        print('@@@## Step 5 Done !')
