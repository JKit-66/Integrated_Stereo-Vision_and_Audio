import time
#import cv2
from picamera2 import Picamera2, Preview
import libcamera

picam2 = Picamera2(0)
picam2.start_preview(Preview.QTGL)   #LEFT
main_stream = {}
lores_stream = {"size": (800, 450)}
my_config = picam2.create_still_configuration(main_stream, lores_stream, transform=libcamera.Transform(vflip=1, hflip=1), display="lores")
picam2.configure(my_config)

picam21 = Picamera2(1)
picam21.start_preview(Preview.QTGL)   #RIGHT
my_config1 = picam21.create_still_configuration(main_stream, lores_stream, transform=libcamera.Transform(vflip=1, hflip=1), display="lores")
picam21.configure(my_config1)

picam2.start()
picam21.start()



#request1 = picam21.capture_request()
#request1.release()
idx = 0

while True:
	request = picam2.capture_request()
	request1 = picam21.capture_request()
	print(idx)
	#request.save("main", f'./pairs/L_CAM/L_{idx}.jpg')
	#request1.save("main", f'./pairs/R_CAM/R_{idx}.jpg')
	idx += 1
	request.release()
	request1.release()
	
	print("Still image captured!")
	time.sleep(2)
	
