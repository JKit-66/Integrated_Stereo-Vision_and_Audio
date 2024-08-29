from ultralytics import YOLO
import cv2
import math
import cvzone

def detect(img):
    results = model(img, stream=True)

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
                        
                        
                #print(conf)
                        
                #class name
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(40,y1)))
                    #print(classNames[cls])
            
            
    cv2.imshow("Image", img)

    cv2.waitKey(1)


if __name__ == "__main__":
    print('Starting...')

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    model = YOLO('training6/best.pt')
    classNames = ["cat","dog","drone", "phone"]
    
    while True:
        success, img  = cap.read()
        img = cv2.flip(img, 1)
        detect(img)
        
        
