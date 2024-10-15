## for yolov8 ##
from ultralytics import YOLO

## train yolov8 ##
## before traing, we need to fix .yaml file, and change the path of train and val data ##
model = YOLO(r'C:\Users\ZJ\Desktop\AI4construciton\detection\yolov8_original.pt')  # load a pretrained model (recommended for training)
model.train(data=r'C:\Users\ZJ\Desktop\AI4construciton\detection\MOCS.yaml', epochs=100, imgsz=640, batchsize = 32)