import time
#import cv2
from picamera2 import Picamera2, Preview
import libcamera
import threading
from datetime import datetime

class take_Vids():
    def __init__(self):
        self.filename = 0
        #self.picam2 = Picamera2(0)
        self.main_stream = {}
        self.lores_stream = {"size": (800, 600)}
        self.barrier = threading.Barrier(2)
        self.stop_event = threading.Event()
        #self.picam2.start_preview(Preview.QTGL)

        #self.my_config = self.picam2.create_video_configuration(self.main_stream, self.lores_stream, transform=libcamera.Transform(vflip=1, hflip=1), display="lores")
        #self.picam2.configure(self.my_config)

        #self.picam21 = Picamera2(1)
        #self.picam21.start_preview(Preview.QTGL)
        #self.my_config1 = self.picam21.create_video_configuration(self.main_stream, self.lores_stream, transform=libcamera.Transform(vflip=1, hflip=1), display="lores")
        #self.picam21.configure(self.my_config1)
    
    def final_rec_vids(self):
        fileN = 0 
        
        thread1 = threading.Thread(target=self.rec_Vids, args=(0, fileN))
        thread2 = threading.Thread(target=self.rec_Vids, args=(1, fileN))
            
        thread1.start()
        thread2.start()
        
        try:
            while True:
                pass
        except KeyboardInterrupt:
            # Stop the threads when you press Ctrl+C
            print("Stopping recording...")
            self.stop_event.set()

            # Wait for the threads to finish
            thread1.join()
            thread2.join()
        
        print("Both videos have been recorded.")
    
    def rec_Vids(self, camera_id, fileN):
        #self.picam2.start()
        #self.picam21.start()
        

        picam2 = Picamera2(camera_id)
        config1 = picam2.create_video_configuration(self.main_stream, self.lores_stream, transform=libcamera.Transform(vflip=1, hflip=1), display="lores")
        picam2.configure(config1)
        
        while not self.stop_event.is_set():
            #current_time = datetime.now()
            #print("Current time Dummy Video:", current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

            try:
                #current_time = datetime.now()
                # Print the current time
                #print("Current time Video start:", current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
                self.barrier.wait()
                picam2.start_and_record_video(f'stereoCollect/keyboard/v{fileN}_{camera_id}.mp4', duration=1)
                #endtime = datetime.now()
                # Print the current time
                #print("Current time Video end:", endtime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
                
                fileN += 1
                self.barrier.wait()
            except KeyboardInterrupt:
                picam2.stop_recording()
                sys.exit()
            
        '''request = picam2.capture_request()
        request1 = picam21.capture_request()
        request.save("main", "R_IMG.jpg")
        request1.save("main", "L_IMG.jpg")
        request.release()
        request1.release()
        print("Still image captured!")'''

if __name__ == '__main__':
    tk_video = take_Vids()
    tk_video.final_rec_vids()
        
    
