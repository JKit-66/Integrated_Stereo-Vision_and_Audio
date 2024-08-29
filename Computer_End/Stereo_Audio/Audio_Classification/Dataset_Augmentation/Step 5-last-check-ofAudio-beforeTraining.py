import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pyaudio
import time
import pandas as pd
from collections import Counter
import soundfile as sf
from pydub import AudioSegment

class batchAnalysis():
    def __init__(self , directt):
        self.directory = directt #'2-trimmedaudio' #'3major_classes'
        self.editable_file = []
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
    
    def get_number_of_channels(self, file_path):
        # Open the audio file
        with sf.SoundFile(file_path) as f:
            # Get number of channels
            num_channels = f.channels
        
        return num_channels
    
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
    editor = batchAnalysis('2-trimmedaudio')
    editor.give_file_direct()
    stat_list = []
    results = []
    start_time = time.time()
    results = []

    now = time.time()
    for j in editor.editable_file:
        #parent_directory = os.path.dirname(j)
        filename = os.path.basename(j)
        filetype = os.path.splitext(filename)[1]

        if editor.indicator == True:
            #print(j)
            try:
                y, sr = librosa.load(j,sr=None)
                num_chan = editor.get_number_of_channels(j)
                classname = os.path.basename(os.path.dirname(j))
                results.append({'Directory': j, 'File': filename, 'Class':classname,  'No of Channels': num_chan})
            except AttributeError as e:
                pass
            else:
                pass
    
    results_df = pd.DataFrame(results)
    #print(results_df)

    now2 = time.time()

    csv_filename = 'Trimmed_audio_details—BeforeTraining.csv'
    csv_file_path = os.path.join(os.path.dirname(editor.directory), 'csv_result', csv_filename)
    results_df.to_csv(csv_file_path, index=False)
    print(now2 - now, 'sec')
    print('@@@## Step 8 Done !')


    csv_file_path = csv_file_path.replace("\\", '/')
    dfnow = pd.read_csv(csv_file_path)
    filtered_df_no_chann = dfnow[(dfnow['No of Channels'] == 2)]
    all_of_stereo = filtered_df_no_chann['Directory'].tolist()

    for audio in all_of_stereo:
        stereo_audio = AudioSegment.from_wav(audio)
        mono_audio = stereo_audio.set_channels(1)
        filename = 'm_' + os.path.basename(audio)
        mono_audio.export(os.path.join(os.path.dirname(audio), filename), format="wav")

        os.remove(audio)

    print('@@@## Step 9 Done !')

    editortoo = batchAnalysis('2-trimmedaudio')
    editortoo.give_file_direct()
    stat_list = []
    results = []
    start_time = time.time()
    results = []

    now = time.time()
    for j in editortoo.editable_file:
        #parent_directory = os.path.dirname(j)
        filename = os.path.basename(j)
        filetype = os.path.splitext(filename)[1]

        if editortoo.indicator == True:
            #print(j)
            try:
                y, sr = librosa.load(j,sr=None)
                num_chan = editortoo.get_number_of_channels(j)
                classname = os.path.basename(os.path.dirname(j))
                spl_now = editor.is_quiet(j)
                spl_too = editor.calculate_spl(j)
                results.append({'Directory': j, 'File': filename, 'Class':classname,  'Duration (s)': round(len(y)/sr,4), 'No of Channels': num_chan, 'Sampling Rate': sr, 'Sound Pressure Level (dB)':spl_now, 'Sound Pressure Level w Reference Pressure (dB)':spl_too})
            except AttributeError as e:
                pass
            else:
                pass
    
    results_df = pd.DataFrame(results)
    #print(results_df)

    now2 = time.time()

    csv_filename = 'Trimmed_audio_details—BeforeTraining.csv'
    csv_file_path = os.path.join(os.path.dirname(editortoo.directory), 'csv_result', csv_filename)
    results_df.to_csv(csv_file_path, index=False)
    print(now2 - now, 'sec')
    
    csv_file_path2 = csv_file_path.replace("\\", '/')
    dfnow2 = pd.read_csv(csv_file_path2)
    df2now_sr = dfnow2['Sampling Rate'].tolist()
    df2now_noC = dfnow2['No of Channels'].tolist()
    df2now_dur = dfnow2['Duration (s)'].tolist()

    if (len(set(df2now_sr)) == len(set(df2now_noC))) and (len(set(df2now_noC)) == 1) and (len(set(df2now_dur))) == 1:
        print('@@@## Step 10 Done !')
    else:
        print('@@@## Something Wrong Step 10 UnDone !')

    
    
    
