"""from roboflow import Roboflow
rf = Roboflow(api_key="wJYMHjporLYxtMW02sxw")
project = rf.workspace("uni-project-1").project("project-1-thfya")
dataset = project.version(1).download("yolov8")
"""

from ultralytics import YOLO
import ultralytics.hub.utils as hub_utils
import matplotlib
#matplotlib.use('TkAgg')

#start_time = time.time()

hub_utils.ONLINE = False

# Load the model.
model = YOLO('/home/jkl1a20/snap/snapd-desktop-integration/157/Desktop/projectfolder/roboflow/yolov8s.pt')
 
# Training.
results = model.train(
   data='/home/jkl1a20/snap/snapd-desktop-integration/157/Desktop/projectfolder/roboflow/TrainData/data.yaml',
   imgsz=800,  
   epochs=25,
   name='finale',
   plots=True
)
