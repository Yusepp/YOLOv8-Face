import torch
import numpy as np
from ultralytics import YOLO
import cv2

def get_model(weights):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(weights)                
    model.to(device)
    return model


def detect_faces(model, imgs, box_format='xyxy',th=0.5):
    total_crops = []
    total_boxes = []
    total_scores = []
    total_cls = []
    
    results = model.predict(imgs, stream=False)  
    
    for i, image_result in enumerate(results):
        filtered_crops = []
        filtered_boxes = []
        filtered_scores = []
        filtered_cls = []
        
        img = imgs[i]
        
        if box_format == 'xyxy':
            image_boxes = image_result.boxes.xyxy.cpu().numpy()
        elif box_format == 'xywh':
            image_boxes = image_result.boxes.xywh.cpu().numpy()
        elif box_format == 'xyxyn':
            image_boxes = image_result.boxes.xyxyn.cpu().numpy()
        elif box_format == 'xywhn':
            image_boxes = image_result.boxes.xywhn.cpu().numpy()
              
        image_scores = image_result.boxes.conf.cpu().numpy()
        image_cls = image_result.boxes.cls.cpu().numpy()
        
        for j in range(len(image_boxes)):
            (x1, y1, x2, y2), score, c = image_boxes[j], image_scores[j], image_cls[j]
            if score >= th:
                filtered_boxes.append([int(x1),int(y1),int(abs(x2-x1)),int(abs(y2-y1))])
                filtered_scores.append(score)
                filtered_cls.append(c)
                filtered_crops.append(img.crop([x1,y1,x2,y2]))
        
        total_crops.append(filtered_crops)        
        total_boxes.append(filtered_boxes)
        total_scores.append(filtered_scores)
        total_cls.append(filtered_cls)
                

    return total_crops, total_boxes, total_scores, total_cls