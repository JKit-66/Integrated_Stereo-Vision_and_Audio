import sys
import pyaudio
#from scipy.io.wavfile import read as scipy_wave_read
from scipy.io import wavfile
import time
from datetime import datetime
import csv
import numpy as np
import timeit
from mic_det import Ears
from datetime import datetime
import wave
import os
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, resample
import pandas as pd

class recorDAUDIO():

    def __init__(self):
            
        self.f_file = []
        #--------global para--------
        self.p = pyaudio.PyAudio()
        self.RATE = 44100
        self.CHANNELS=1
        self.CHUNK =  4 #4
        #self.WIDTH = 2
        self.FORMAT = pyaudio.paInt16
        self.duration = 1  # seconds
        self.filename_counter = 0
        self.stream_list = []
        self.buffer_size = 200 #200
        self.delimt = '@#@_'
        #--------global para--------
        # 0 is Left, 1 is Right
        
        
    def callback(self, in_data, frame_count, time_info, status):
        return (in_data, pyaudio.paContinue)

    def makeStream(self, INDEX):
        stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        input_device_index = INDEX,
                        frames_per_buffer=self.CHUNK)
        return stream

    def get_usb_info(self):
        usb_port= []
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        for i in range(0, numdevices):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                self.check_dev = list(self.p.get_device_info_by_host_api_device_index(0, i).get('name'))
                if self.check_dev[0] == 'U' and self.check_dev[1] == 'S' and self.check_dev[2] == 'B':
                    usb_port.append(i)
        return usb_port

    def record_utterance(self):
        current_time = datetime.now()
        #print("Current time Dummy Audio:", current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        
        frames = [ [] for _ in range(len(self.stream_list))]
        audio_in_np = [ [] for _ in range(len(self.stream_list))]
        #start = timeit.default_timer()
        time_stp = [[] for _ in range(len(self.stream_list))]
        self.f_file = []
        init_time = time.perf_counter()
        #init_time = time.process_time()
        
        for i in range(0, int(self.RATE / self.CHUNK * self.duration)):
            for j in range(len(self.stream_list)):
                
                if len(frames[j]) < 1:
                    current_time = datetime.now()
                    #print("Current time Audio Start:", current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
                    
                data = self.stream_list[j].read(self.CHUNK, exception_on_overflow = False)
                #curr_time = time.perf_counter() - init_time #time.process_time() - init_time #  #
                #numpydata = np.frombuffer(data, dtype=np.int16)
                    
                frames[j].append(data)
                #stop1 = timeit.default_timer()
                #time_stp[j].append(curr_time)
                #numpies[j].append(numpydata)
            #print(numpies)
        stop = time.perf_counter() #timeit.default_timer()
        #print('Finished: ', stop - init_time)
        
        #start1 = timeit.default_timer()
        for i in range(len(self.stream_list)):
            current_time = datetime.now()
            #print("Current time Audio End:", current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        
            filename = 'stereoCollect/keyboard/' + 'a_'+ str(self.filename_counter)+ f'_{i}' + '.wav'
            #print(filename)
            self.f_file.append(filename)
            wf = wave.open(os.path.join(filename), 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames[i]))
            wf.close()
            '''
            bufferZ = io.BytesIO()
            wf = wave.open(bufferZ, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames[i]))
            wf.close()
            
        
            bufferZ.seek(0)  # Reset buffer position to the beginning
            wf = wave.open(bufferZ, 'rb')
            audio_data = wf.readframes(wf.getnframes())
            audio_in_np[i].append(np.frombuffer(audio_data, dtype=np.int16))

            mo.append(time.perf_counter() - init_time)
            curr3_time = time.perf_counter() - init_time
            stop1 = timeit.default_timer()
            print(np.diff(mo))
        
        for ij in range(len(self.stream_list)):
            filename = 'z' + str(self.filename_counter)+ f'_{ij}' + '.txt'
            wf = wave.open(os.path.join(filename), 'wb')
            wf.writeframes(b''.join(time_stp[ij]))
            wf.close()

        for j in range(len(stream_list)):
            stream_list[j].stop_stream()
        '''
        self.filename_counter += 1
        #return audio_in_np
    
    def read_AUDIO(self, filename):
        #filename = '0_0.wav'
        sample_rate, sample_data = wavfile.read(filename)
        print(sample_data)

    
    def audio_trimm(self, fileN):
        starting_path=''
        files = os.path.join(starting_path, fileN)
        rate, data = wavfile.read(files)
        return np.array(data)
    
    def start_recording_now(self):
        audio_det = Ears()
        usb_num = self.get_usb_info()
        choices = 1
        res = []
        
        for idx in range(len(usb_num)):              
            self.stream_list.append(self.makeStream(usb_num[idx]))
        
        time.sleep(2)
        
        while True:
            try:
                print('Starting...')
                
                self.record_utterance()
                #audio_L_np, audio_R_np = audio_np[0], audio_np[1] 
                #time.sleep(1.1)
                #print(self.f_file)
                R, L = self.audio_trimm(self.f_file[0]), self.audio_trimm(self.f_file[1])
                R = R * 0.941266
                #print(recordING.f_file, recordING.g_file)
                #g , f = audio_L_np, audio_R_np
                
                ###Time Delay###
                time_delay = self.check_delay(L, R)  #R,L
                #print('Before Correction:', time_delay)
                corrected_time_delay = time_delay - (-0.004056)
                #print('After Correction:', corrected_time_delay)
                sound_dir = audio_det.direction_estimation(corrected_time_delay)
                
                ###Comapre A,plitude to find Direction###
                spl_R, max_R = audio_det.calculate_spl(R)
                spl_L, max_L = audio_det.calculate_spl(L)
                
                sound_L_R, pk_L_R = audio_det.compare_amplitudes(max_L, max_R)
                
                average_spl =  (spl_R + spl_L)/ 2
                sound_dist = audio_det.distance_estimation(average_spl)
                print(self.delimt + str(sound_dir) + '_' + sound_L_R + '_' + str(sound_dist))
                
                currDT = datetime.now()
                res.append({'Time_Now': currDT.strftime("%H:%M:%S"),'L_F': self.f_file[1], 'R_F': self.f_file[0], 'L_SPL':spl_L, 'R_SPL':spl_R, 'Max_SPL_L': max_L, 'Max_SPL_R': max_R, 'PeakRatio_L_R': pk_L_R, 'Time_Delay': corrected_time_delay, 'Sound_Dir': sound_dir, 'Loud?_LR':sound_L_R, 'Sound_Dist':sound_dist})
            
            except KeyboardInterrupt:
                results_df = pd.DataFrame(res)
                csv_file_path = 'csv_result/12_08_Audio_L_R_direction.csv'
                results_df.to_csv(csv_file_path, index=False)
                for i in range(number_of_mics):
                    stream_list[i].close()
                sys.exit()


    def check_delay(self, g, f):
        fs = 44100  # Sampling frequency in Hz
        s1 = g
        s2 = f
        correlation = fftconvolve(s1, s2[::-1], mode='full')

        lags = np.arange(-len(s1) + 1, len(s1))
        max_lag = lags[np.argmax(correlation)]
        #print(max_lag)
        # Convert lag to time delay
        time_delay = max_lag / fs
        return (round(time_delay, 8))


            


if __name__ == '__main__':
    #getDeviceInfo()
                
    '''def getDeviceInfo():
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                n = p.get_device_info_by_host_api_device_index(0, i).get('name')
                print("Input Device id ", i,"-", n.encode("utf8").decode("cp950", "ignore"))'''
    
    recordING = recorDAUDIO()
    
    usb_num = recordING.get_usb_info()
    choices = 1
    
    res = []
    
    recordING.start_recording_now()
    
    '''
    for idx in range(len(usb_num)):              
        recordING.stream_list.append(recordING.makeStream(usb_num[idx]))

    if choices == 1:
        time.sleep(2)
        print("""


 ░▒▓███████▓▒░ ░▒▓████████▓▒░  ░▒▓██████▓▒░  ░▒▓███████▓▒░  ░▒▓████████▓▒░ 
░▒▓█▓▒░           ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░    ░▒▓█▓▒░     
░▒▓█▓▒░           ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░    ░▒▓█▓▒░     
 ░▒▓██████▓▒░     ░▒▓█▓▒░     ░▒▓████████▓▒░ ░▒▓███████▓▒░     ░▒▓█▓▒░     
       ░▒▓█▓▒░    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░    ░▒▓█▓▒░     
       ░▒▓█▓▒░    ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░    ░▒▓█▓▒░     
░▒▓███████▓▒░     ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░    ░▒▓█▓▒░     
                                                                           

 """)
        while True:
            try:
                print('Starting...')
                recordING.record_utterance()
                #audio_L_np, audio_R_np = audio_np[0], audio_np[1] 
                #time.sleep(1.1)
                print(recordING.f_file)
                R, L = recordING.audio_trimm(recordING.f_file[0]), recordING.audio_trimm(recordING.f_file[1])
                R = R * 0.941266
                #print(recordING.f_file, recordING.g_file)
                #g , f = audio_L_np, audio_R_np
                
                ###Time Delay###
                time_delay = recordING.check_delay(L, R)  #R,L
                print('Before Correction:', time_delay)
                corrected_time_delay = time_delay - (-0.004056)
                print('After Correction:', corrected_time_delay)
                sound_dir = audio_det.direction_estimation(corrected_time_delay)
                
                ###Comapre A,plitude to find Direction###
                spl_R, max_R = audio_det.calculate_spl(R)
                spl_L, max_L = audio_det.calculate_spl(L)
                
                sound_L_R, pk_L_R = audio_det.compare_amplitudes(max_L, max_R)
                print('>>>>>', sound_dir, sound_L_R)
                
                
                average_spl =  (spl_R + spl_L)/ 2
                sound_dist = audio_det.distance_estimation(average_spl)
                print('||@||', sound_dist)
                
                currDT = datetime.now()
                res.append({'Time_Now': currDT.strftime("%H:%M:%S"),'L_F': recordING.f_file[1], 'R_F': recordING.f_file[0], 'L_SPL':spl_L, 'R_SPL':spl_R, 'Max_SPL_L': max_L, 'Max_SPL_R': max_R, 'PeakRatio_L_R': pk_L_R, 'Time_Delay': corrected_time_delay, 'Sound_Dir': sound_dir, 'Loud?_LR':sound_L_R, 'Sound_Dist':sound_dist})
                
                
                
            except KeyboardInterrupt:
                results_df = pd.DataFrame(res)
                csv_file_path = 'csv_result/12_08_Audio_L_R_direction.csv'
                results_df.to_csv(csv_file_path, index=False)
                for i in range(number_of_mics):
                    stream_list[i].close()
                sys.exit()'''
    
    if choices == 2:
        recordING.read_AUDIO('0_0.wav')
