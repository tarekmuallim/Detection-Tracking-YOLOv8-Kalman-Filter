# Detection-Tracking-YOLOv8-Kalman-Filter
Bolts and nuts detection and tracking using yolov8 and kalman tracker.

Install:
```
pip install filterpy
pip install supervision
pip install pylabel
pip install ultralytics
sudo apt install libopencv-dev python3-opencv
```
Train:

`yolo task=detect mode=train model=yolov8n.pt data='dataset/stromaDataset.yaml' batch=64 epochs=20 imgsz=640
`

Live demo:

`python3 tracking_demo.py -m 'best.pt' -vs 'original_dataset/videos/val/val.mp4' -vo 'res.mp4' -s 0`
