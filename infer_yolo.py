import os

# Inference command on test images
# Replace 'best.pt' with path to trained weights
os.system("python yolov5/detect.py --weights runs/train/sesame_weed_detection/weights/best.pt --img 512 --conf 0.4 --source data/test_images/")


