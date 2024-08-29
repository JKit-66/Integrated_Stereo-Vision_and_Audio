import os
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

#print(cv2.getBuildInformation())
#(4608, 2592)
# Global variables preset
total_photos = 35







img_width = 4608 #800

img_height = 2592 #450
photo_width = img_width * 2
photo_height = img_height #450
image_size = (img_width,img_height)

# Chessboard parameters
rows = 6
columns = 8
square_size = 2.75


calibrator = StereoCalibrator(rows, columns, square_size, image_size)
photo_counter = 5








print ('Start cycle')

while photo_counter != total_photos:
  photo_counter = photo_counter + 1
  print ('Import pair No ' + str(photo_counter))
  leftName = './pairs/L_CAM/L_'+str(photo_counter).zfill(2)+'.jpg'
  rightName = './pairs/R_CAM/R_'+str(photo_counter + 1).zfill(2)+'.jpg'
  
  if os.path.isfile(leftName) and os.path.isfile(rightName):
      imgLeft = cv2.imread(leftName,1)
      imgRight = cv2.imread(rightName,1)
      try:
        calibrator._get_corners(imgLeft)
        calibrator._get_corners(imgRight)
        print('Done')
      except ChessboardNotFoundError as error:
        print (error)
        print ("Pair No "+ str(photo_counter) + " ignored")
      else:
        calibrator.add_corners((imgLeft, imgRight), True)
        
print ('End cycle')


print ('Starting calibration... It can take several minutes!')
calibration = calibrator.calibrate_cameras()
calibration.export('calib_result')
print ('Calibration complete!')



# Lets rectify and show last pair after  calibration
calibration = StereoCalibration(input_folder='calib_result')
rectified_pair = calibration.rectify((imgLeft, imgRight))

cv2.imshow('Left CALIBRATED', rectified_pair[0])
cv2.imshow('Right CALIBRATED', rectified_pair[1])
cv2.imwrite("rectifyed_left.jpg",rectified_pair[0])
cv2.imwrite("rectifyed_right.jpg",rectified_pair[1])
cv2.waitKey(0)
