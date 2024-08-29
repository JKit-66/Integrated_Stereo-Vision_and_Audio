import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import seaborn as sns
import pathlib
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import random
import wave
from scipy.io import wavfile 
from scipy.fftpack import fft
from collections import Counter
#import seaborn as sns
#from memory_profiler import profile
import time
from pydub import AudioSegment
import pandas as pd

imported = tf.saved_model.load("C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioWeightTraining_Testing/rsflow_result/tensorflowSaved")

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

# Convert waveform to spectrogram
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram[..., tf.newaxis]

# Creating spectrogram dataset from waveform or audio data
def get_spectrogram_dataset(dataset):
    dataset = dataset.map(
        lambda x, y: (get_spectrogram(x), y),
        num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def obtain_res(input_data):
    #print(res['class_names'])
    predicted_res = list(input_data['predictions'][0]._numpy())
    #print(input_data['predictions'])
    #sec_res = predicted_res
    #print(predicted_res)
    #print(predicted_res)
    max_res = max(predicted_res)
    #sec_res.remove(max_res)
    #sec_max_res = max(sec_res)
    #print(predicted_res)

    result_book = {
        0:"bellphone",
        1:"drone",
        2:"iphone",
        3:"jogging",
        4:"keyboard",
        5:"silence",
        6:"telephone"}
    
    if max_res >= 0.4:
      index = predicted_res.index(max_res)
      #index2 = predicted_res.index(sec_max_res)
      conf = max_res
      return result_book[index], conf

    else:
      res = 'ERROR'
      conf = max_res
      return res, conf



def options_AUDIO_within_file_categories(file_directory):
    res_csv = []
    finale_ = []
    choices = ["jogging", "iphone", "keyboard", "bellphone", "drone", "telephone"]
    init_num = 0
    #get audio file time length
    directory = file_directory
    num_cat = len(choices)

    for i in range(num_cat):
        #cat = next(os.walk(directory))[1][i]
        pre_cat = choices[i]
        cat = choices[i] 
        print(cat)

        ans = os.listdir(str(directory + '/' + cat))
        files2 = len(ans)
        init_num += files2

        for j in range(files2):
            directory2 = str(directory) + "/{}".format(cat)
            cat2 = next(os.walk(directory2))[2][j]
            audio_path  = os.listdir(str(directory+ '/' + cat))
            path = os.path.join(directory, cat, audio_path[j])

            Input = tf.io.read_file(str(path))
            x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=44100,)
            audio, labels = squeeze(x, 'yes')
            waveform = audio
            res = imported(waveform[tf.newaxis, :])
            clssZ_det = str(res['class_names'][0]._numpy())[2:-1]
            obtain, conf_scr = obtain_res(res)

            finale_.append(f'{pre_cat}:{obtain}')
            res_csv.append({'File_Dir':path, 'FileName':os.path.basename(path), 'Thr_Cls':cat, 'Det_Cls': obtain, 'Second_Det_Cls': clssZ_det, 'Conf_Scr': conf_scr})


    results_df = pd.DataFrame(res_csv)
    csv_file_path = 'C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioWeightTraining_Testing/csv_res/13-08_audio_classification.csv'
    results_df.to_csv(csv_file_path, index=False)

    print(Counter(finale_))
    print(sum(Counter(finale_).values()))

def options_AUDIO_uncatogerized(file_directory):
  directory = file_directory
  ans = os.listdir(str(directory))
  files2 = len(ans)
  finale_= []
  results = []
  for j in range(files2):
    directory2 = str(directory)
    cat2 = next(os.walk(directory2))[2][j]
    audio_path  = os.listdir(str(directory+ '/'))
    path = os.path.join(directory, audio_path[j])
    
    Input = tf.io.read_file(str(path))
    x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=44100,)
    audio, labels = squeeze(x, 'yes')
    waveform = audio
    res = imported(waveform[tf.newaxis, :])
    obtain, _ = obtain_res(res)

    finale_.append(f'{audio_path[j]}:{obtain}')
    results.append(obtain)

  print(Counter(finale_))
  print(Counter(results))  
  
 

if __name__ == "__main__":
    options_AUDIO_within_file_categories('C:/Users/J Kit/Desktop/Y3 Intern/Week6-AudioWeightTraining_Testing/validation_dataset')
    #options_AUDIO_uncatogerized('audioFOLDER/virtENV/onesec/dog')
    #options_AUDIO_amplify('audioFOLDER/virtENV/stop', 4)
    #plot_fft_logscale('audioFOLDER/virtENV/stop','stop')
    
            


        
    
