import cv2
import numpy as np
import threading
import cv2
import cvzone
import math
from picamera2 import Picamera2, Preview

'''
class Start_Cameras:

    def __init__(self, sensor_id):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

        self.sensor_id = sensor_id

        gstreamer_pipeline_string = self.gstreamer_pipeline()
        self.open(gstreamer_pipeline_string)

    #Opening the cameras
    def open(self, gstreamer_pipeline_string):
        gstreamer_pipeline_string = self.gstreamer_pipeline()
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            grabbed, frame = self.video_capture.read()
            print("Cameras are opened")

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)
            return
        # Grab the first frame to start the video capturing
        self.grabbed, self.frame = self.video_capture.read()

    #Starting the cameras
    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera, daemon=True)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()

    # Currently there are setting frame rate on CSI Camera on Nano through gstreamer
    # Here we directly select sensor_mode 3 (1280x720, 59.9999 fps)
    def gstreamer_pipeline(self,
            sensor_mode=3,
            capture_width=1280,
            capture_height=720,
            display_width=640,
            display_height=360,
            framerate=30,
            flip_method=0,
    ):
        return (
                "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    self.sensor_id,
                    sensor_mode,
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
        )

'''

class Start_Cameras:

    def __init__(self, sensor_id):
        self.main_size = (640, 320) #(4608, 2592) #(640, 320)
        self.main_format = "RGB888"
        self.cam_id = sensor_id
        self.frame = None
        self.grabbed = False
        self.read_lock = threading.Lock()
        self.running = False
        self.video_capture = None
    
    def open(self):
        try:
            self.video_capture = Picamera2(self.cam_id)
            self.video_capture.preview_configuration.main.size=self.main_size
            self.video_capture.preview_configuration.main.format=self.main_format
            self.video_capture.preview_configuration.main.align()
            self.video_capture.start()
            img = self.video_capture.capture_array()
            img = cv2.flip(img, 0)
            grabbed, img = self.video_capture.read()
            print("Cameras are opened")

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + self.cam_id)
            return
        # Grab the first frame to start the video capturing
        self.grabbed, self.frame = self.video_capture.read()
        
    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera, daemon=True)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
                
                
    def read(self):
        with self.read_lock:
            img = self.frame.copy()
            grabbed = True
            
        return grabbed, img
        
        
#This is the main. Read this first. 
if __name__ == "__main__":
    main_size = (640, 320) #(4608, 2592) #(320, 240)
    main_format = "RGB888"
    left_grabbed = True
    right_grabbed = True
    
    piCam_a= Picamera2(1)
    piCam_b= Picamera2(0)
    picam = [piCam_a, piCam_b]
    
    for cam in picam:
        cam.preview_configuration.main.size=main_size
        cam.preview_configuration.main.format=main_format
        cam.preview_configuration.main.align()
        cam.start()
        
    while True:
        left_frame = piCam_a.capture_array()
        left_frame = cv2.flip(left_frame, -1)

        #right_frame = left_frame
        right_frame = piCam_b.capture_array()
        right_frame = cv2.flip(right_frame, -1)
        
        if left_grabbed and right_grabbed:
            images = np.hstack((left_frame, right_frame))
            cv2.imshow("Camera Images", images)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                cv2.imwrite('./scenes/photo_7.jpg', images) 
                break
        else:
            break
    
    
    cv2.destroyAllWindows()