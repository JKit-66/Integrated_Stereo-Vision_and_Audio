from ultralytics import YOLO
import cv2
import math
import cvzone
from collections import Counter
import sounddevice as sd
import scipy.io.wavfile as sp
import wavio as wv
import datetime
import uuid
import time
import numpy as np
import math
from idlecolors import *

import tensorflow as tf

class sensorM():

    def __init__(self):
        # Sampling frequency
        self.freq = 44100
        # Recording duration
        self.duration = 1

        self.audio_model = ('audioFOLDER/tensorflowSaved')
        self.imported = tf.saved_model.load(self.audio_model)

        
        self.directory = 'audioFOLDER/forFinalDetections/'
        self.file2analyze = ''

        self.microP_res = {}

        self.audio_result_book = {"bellphone":0,
                             "drone":0,
                             "iphone":0,
                             "shoe":0,
                             "keyboard":0,
                             "silence":0,
                             "telephone":0,
                             "background":0
                             }

        self.new_audio_result_book = {"drone":0,
                                  "phone":0,
                                  "keyboard":0,
                                  "shoe":0,
                                  "background":0}

        '''self.audio_result_book = {'cat':0,
                             'dog':0,
                             'drone':0,
                             'phone':0,
                             'background':0
                             }'''
        
    def merge_clss(self):
         old_ = self.audio_result_book
         phones_to_merge = ['bellphone', 'iphone', 'telephone']
         highest_value = max(old_[key] for key in phones_to_merge)
         new_dict = {
            'phone': highest_value,
            **{k: v for k, v in old_.items() if k not in phones_to_merge}}
         silence_value = new_dict['silence']

         if silence_value > 0.45:
             new_dict['background'] = silence_value

         new_dict.pop('silence')
         self.new_audio_result_book = new_dict
        
        


        
    def random_name_generator(self):
        now = datetime.datetime.now()
        tod = datetime.date.today()
        HH = now.hour
        mm = now.minute

        MM, DD = tod.month, tod.day
        unique = str(uuid.uuid4())
        unique = unique.split('-')[0]
        fileN = f'{MM}{DD}_{HH}{mm}_{unique}'
        return fileN

    def record_ON(self, name):
        print("#############Starting Listening...#################")
        recording = sd.rec(int(self.duration * self.freq), 
                           samplerate=self.freq, channels=1)

        sd.wait()
        filename = self.directory + f'{name}.wav'
        wv.write(filename, recording, self.freq, sampwidth=2)

        self.file2analyze = filename
        

    def record_for_process(self):
        fileN = self.random_name_generator()
        self.record_ON(fileN)
        print("---------------------------------------------------")
        print('#############Finish Listening.....#################')

    def obtain_res(self,input_data):
        self.microP_res = {}
        predicted_res = list(input_data['predictions'][0]._numpy())
        #return input_data['predictions']
        predicted_res.append(0)
        max_res = max(predicted_res)

        result_book = {
                    0:"bellphone",
                    1:"drone",
                    2:"iphone",
                    3:"shoe",
                    4:"keyboard",
                    5:"silence",
                    6:"telephone",
                    7:"background"}

        self.audio_result_book = {"bellphone":0,
                             "drone":0,
                             "iphone":0,
                             "shoe":0,
                             "keyboard":0,
                             "silence":0,
                             "telephone":0,
                             "background":0
                             }
        
        
        #clss_merger = ["bellphone", "iphone", "telephone" ]
        #to_phone = {"bellphone":0, "iphone": 0 , "telephone":0}
        if max_res >= 0.4:
          for value, key in enumerate(self.audio_result_book.items()):
            self.audio_result_book[key[0]] = predicted_res[value]
                

        else:                
            self.audio_result_book["background"] = 1


        self.merge_clss()
        #print(self.audio_result_book)
            
          


    def squeeze(self, audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels

    def microP_READ(self):
        audio_INPUT = self.file2analyze

        #print(audio_INPUT)
        Input = tf.io.read_file(audio_INPUT)
        
        x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=44100,)
        #print(x)
        waveform, labels = self.squeeze(x, 'yes')
        res = self.imported(waveform[tf.newaxis, :])
        self.obtain_res(res)
        #print(self.microP_res)
        #print('::', self.audio_result_book)
        
        
    def process_detection_res(self, inputs):
        if (len(inputs) > 0): 
            max_conf = max(inputs, key=inputs.get)
            return max_conf, inputs[max_conf]

        else:
            return None, None
    
    def record_and_process(self):
        print(self.file2analyze)



class sensorC():

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.model = YOLO('visionFOLDER/trainingfile3/best.pt')
        self.classNames = ["drone","phone","keyboard", "shoe"]

        #result
        self.cam_result_book = {} #{'cat':0.6, 'dog':0.9}
        self.cam_conf = 0

        self.empty = 0
    
    def detect_cam(self, img):
        self.cam_result_book = {"drone":0,
                                "phone":0,
                                "keyboard":0,
                                "shoe":0,
                                "background":0
                                }
        counter = 0
        results = self.model(img, verbose=False, stream=True)
        for res in results:
            boxes = res.boxes

            if (len(boxes.cls)) == 0:
                self.cam_result_book["background"] = 1

            else:

                for box in boxes:

                    #print(counter)
                    
                    #confidence
                    conf = math.ceil((box.conf[0]*100))/100
                    
                    if conf > 0.50:       
                            #Bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                                
                        w, h = x2-x1, y2-y1
                        bbox = (x1, y1, w, h)
                        cvzone.cornerRect(img, bbox)
                                                            
                        #print(conf)
                                
                        #class name
                        cls = int(box.cls[0])
                        cvzone.putTextRect(img, f'{self.classNames[cls]} {conf}', (max(0,x1), max(40,y1)))
                            #print(classNames[cls])

                        self.cam_result_book[self.classNames[cls]] = conf

                    else:
                        self.cam_result_book["background"] = 1

                
        cv2.imshow("Image", img)

        cv2.waitKey(1)


    def camera_detection(self):
        success, img  = self.cap.read()
        img = cv2.flip(img, 1)
        self.detect_cam(img)


class get_Out_of_the_BEST:

    def __init__(self):
        self.overall = {}
        #self.audio_Dict = audio_D
        #self.vision_Dict = vision_D
            
    def overall_F(self, audio, vision):
        k_audio = 0.4
        k_vision = 1-k_audio
        return (k_vision*vision + k_audio*audio)


    def organize_final_DICT(self, audio_Dict, vision_Dict):
        if (len(audio_Dict)) == (len(vision_Dict)):
            for key in audio_Dict.keys():
                calculated = self.overall_F(audio_Dict[key], vision_Dict[key])
                self.overall[key] = calculated
            
            return (self.overall)


    
    
        


if __name__ == "__main__":
    print('Start Detecting...')
    listening = sensorM()
    looking = sensorC()
    finale = get_Out_of_the_BEST()

    #printc( "If you add " + red("red") + " to " + blue("blue") + ", you get " + purple("purple") + orange(' ok !'))

    while True:
        
        listening.record_for_process()
        looking.camera_detection()
        time.sleep(0.1)
        listening.microP_READ()

        vision_K = looking.cam_result_book
        audio_K = listening.new_audio_result_book

        print(vision_K, audio_K)

        crave_for = finale.organize_final_DICT(audio_K, vision_K)
        print(crave_for)
        one_and_only = max(crave_for, key=crave_for.get)

        print('Visual : ', vision_K)
        print('Audio : ', audio_K)
        print('Integrate Both: ', crave_for)
        
        ans1 = one_and_only
        ans2 = crave_for[one_and_only]
        printc(orange(str(ans1)) + " => " + orange(str(ans2)))
        
        
        
        
        
        
