from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("yolov8s.pt")  # Load model once

def detect_people(frame: np.ndarray):
    results = model(frame)
    detections = results[0].boxes
    people_coords = []
    people_count = 0

    for det in detections:
        cls = int(det.cls[0])
        if cls == 0:  # Person class
            x_center = int(det.xywh[0][0])
            y_center = int(det.xywh[0][1])
            people_coords.append((x_center, y_center))
            people_count += 1

    return people_coords, people_count
