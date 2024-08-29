from ultralytics import YOLO
import cv2
import cvzone
import math

#cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
#cap.set(3, 640)
#cap.set(4, 480)

from picamera2 import Picamera2, Preview

piCam_a= Picamera2(0)
piCam_b= Picamera2(1)
picam = [piCam_a, piCam_b]

'''
for cam in picam:
    cam.preview_configuration.main.size=(640,480)
    cam.preview_configuration.main.format="RGB888"
    cam.preview_configuration.main.align()
    cam.configure('preview')
    cam.start()

'''
piCam_a.preview_configuration.main.size=(800, 450)
piCam_a.preview_configuration.main.format="RGB888"
piCam_a.preview_configuration.main.align()
piCam_a.configure('preview')
piCam_a.start()


model = YOLO('trainingfile3/best.pt')

classNames = ['drone', 'phone', 'keyboard', 'shoe']#["cat","dog","drone", "phone"]

while True:
    img = piCam_a.capture_array()
    img = cv2.flip(img, 0) 
    results = model(img, stream=True, imgsz=[320,416])
    
    for i in results:
        boxes = i.boxes
        
        for box in boxes:
            #confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            if conf > 0.40:
                #Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                
                w, h = x2-x1, y2-y1
                bbox = (x1, y1, w, h)
                cvzone.cornerRect(img, bbox)
                
                #class name
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(40,y1)))
                #print(classNames[cls])
    
    
    cv2.imshow('picam',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()


