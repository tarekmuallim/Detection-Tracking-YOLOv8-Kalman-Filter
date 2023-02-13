import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator

from sort import Sort as kalman_trakcer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, help="YOLO v8 model path", default='best.pt')
parser.add_argument('-vs', '--video_path', type=str, help="Input video path", default='original_dataset/videos/val/val.mp4')
parser.add_argument('-s', '--save', type=int, help="Save result video", default=0)
parser.add_argument('-vo', '--video_output_path', type=str, help="Output video path", default='res.mp4')
args = parser.parse_args()

model_path = args.model_path
video_path = args.video_path
save = args.save
video_output_path = args.video_output_path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)

model = YOLO(model_path)
model.fuse()
model.overrides['degrees'] = 0 #some bug in cfg!!!

CLASS_NAMES_DICT = model.model.names
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=2, text_scale=1)

def plot_bboxes(xyxy, track_id, class_id, frame):     

    detections = Detections(
                xyxy=xyxy,
                confidence=track_id,
                class_id=class_id,
                )    

    labels = [f"{CLASS_NAMES_DICT[class_id]} {confidence:d}"
    for _, confidence, class_id, tracker_id
    in detections]
    
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    
    return frame

def get_res(results):
    confidence=results[0].boxes.conf.cpu().numpy()
    res = [idx for idx, val in enumerate(confidence) if val > 0.7]        

    boxes=results[0].boxes.xyxy.cpu().numpy()
    scores=results[0].boxes.conf.cpu().numpy()
    classes=results[0].boxes.cls.cpu().numpy().astype(int)

    boxes=boxes[res]
    classes=classes[res]
    scores=scores[res]

    return boxes, classes, scores

cap = cv2.VideoCapture(video_path)
assert cap.isOpened()

fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))   
size = (frame_width, frame_height)

if save:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, size)


tracker = kalman_trakcer(max_age=20, min_hits=1, iou_threshold=0.3)

count = 0
while True:
    
    start_time = time()
    
    ret, frame = cap.read()
    assert ret
    
    results = model(frame) 
    boxes, classes, scores = get_res(results)

    boxes1 = boxes
    classes1 = classes
    scores1 = scores

    boxes2 = boxes
    classes2 = classes
    scores2 = scores 

    idxs1 = np.where(classes == 0)[0]#class 1
    idxs2 = np.where(classes == 1)[0]#class 2

    class_1_detections = len(idxs1)
    class_2_detections = len(idxs2)

    if class_1_detections:
        boxes1 = boxes[idxs1]
        scores1 = scores[idxs1]
        classes1 = classes[idxs1]
    else:
        boxes1 = np.empty((0, 5))
        scores1 = scores[idxs1]            

    dets1 = np.hstack((boxes1, scores1[:,np.newaxis]))
    res1 = tracker.update(dets1)

    if class_2_detections:
        boxes2 = boxes[idxs2]
        scores2 = scores[idxs2]
        classes2 = classes[idxs2]
    else:
        boxes2 = np.empty((0, 5))
        scores2 = scores[idxs2] 

    dets2 = np.hstack((boxes2, scores2[:,np.newaxis]))
    res2 = tracker.update(dets2)

    boxes_track1 = res1[:,:-1]
    boces_ids1 = res1[:,-1].astype(int)
    classes1 = np.zeros(boces_ids1.shape, dtype=int)
    count = max(np.append(boces_ids1, count))

    boxes_track2 = res2[:,:-1]
    boces_ids2 = res2[:,-1].astype(int)
    classes2 = np.ones(boces_ids2.shape, dtype=int)
    count = max(np.append(boces_ids2, count))

    if class_1_detections:
        frame = plot_bboxes(boxes_track1, boces_ids1,  classes1, frame)
    if class_2_detections:
        frame = plot_bboxes(boxes_track2, boces_ids2,  classes2, frame)
    
    end_time = time()
    fps = 1/np.round(end_time - start_time, 2)             
    cv2.putText(frame, f'FPS: {int(fps)}  Count: {int(count)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    
    if save:
        video_writer.write(frame)

    cv2.imshow('YOLOv8 Detection', frame) 
    if cv2.waitKey(5) & 0xFF == 27:                
        break

cap.release()
if save:
    video_writer.release()
cv2.destroyAllWindows()