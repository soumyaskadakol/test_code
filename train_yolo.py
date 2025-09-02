import os

# Training command (runs YOLOv5 training)
# Assumes dataset.yaml is already prepared with sesame crop/weed classes
os.system("python yolov5/train.py --img 512 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt --name sesame_weed_detection")
