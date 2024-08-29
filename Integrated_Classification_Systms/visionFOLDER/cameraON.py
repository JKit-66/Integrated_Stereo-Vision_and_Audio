from ultralytics import YOLO
import cv2
import math
import cvzone
from collections import Counter


class sensorX():

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.model = YOLO('C:/Users/J Kit/Desktop/Y3/IP March April/testMERGER/visionFOLDER/training7/best.pt')
        self.classNames = ['drone', 'phone', 'mobile', 'keyboard', 'shoe']

        #result
        self.cam_result_book = {} #{'cat':0.6, 'dog':0.9}
        self.cam_conf = 0

        self.empty = 0
    
    def detect_cam(self, img):
        self.cam_result_book = {'cat':0,
                                'dog':0,
                                'drone':0,
                                'phone':0,
                                'background':0
                                }
        counter = 0
        results = self.model(img, verbose=False, stream=True)
        for res in results:
            boxes = res.boxes

            if (len(boxes.cls)) == 0:
                self.cam_result_book['background'] = 1

            else:

                for box in boxes:

                    print(counter)
                    
                    #confidence
                    conf = math.ceil((box.conf[0]*100))/100
                    
                    if conf > 0.60:       
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
                        self.cam_result_book['background'] = 1

            
                
                    
                
                
        cv2.imshow("Image", img)

        cv2.waitKey(1)


    def camera_detection(self):
        success, img  = self.cap.read()
        img = cv2.flip(img, 1)
        self.detect_cam(img)


    def process_detection_res(self, inputs):
        if (len(inputs) > 0): 
            max_conf = max(inputs, key=inputs.get)
            return max_conf, inputs[max_conf]

        else:
            return None, None
        #print(max_conf)
        #return((Counter(inputs)).most_common(1)[0][0])
        


if __name__ == "__main__":
    print('Starting...')
    cam_sense = sensorX()

    #out = cam_sense.process_detection_res(cam_sense.cam_res)
    #print(out)
    while True:
        cam_sense.camera_detection()
        print(cam_sense.cam_result_book)
        #vid_class, vid_conf = cam_sense.process_detection_res(cam_sense.cam_res)
        
        
        
        
