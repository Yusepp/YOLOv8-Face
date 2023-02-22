import torch
import numpy as np
from ultralytics import YOLO
import cv2

def get_model(weights):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else device
    
    model = YOLO(weights)                
    model.to(device)
    return model


def detect_faces(model, imgs, box_format='xyxy',th=0.5):
    total_crops = []
    total_boxes = []
    total_scores = []
    total_cls = []
    
    results = model.predict(imgs, stream=False, verbose=False)
    
    for i, image_result in enumerate(results):
        filtered_crops = []
        filtered_boxes = []
        filtered_scores = []
        filtered_cls = []
        
        img = imgs[i]
        img_w, img_h = np.array(img).shape[:2]
        
        if box_format == 'xyxy':
            image_boxes = image_result.boxes.xyxy.cpu().numpy()
        elif box_format == 'xywh':
            image_boxes = image_result.boxes.xyxy.cpu().numpy()
            image_boxes = [[round(x1), round(y1), round(np.abs(x1-x2)), round(np.abs(y1-y2))] for x1,y1,x2,y2 in image_boxes]
        elif box_format == 'xyxyn':
            image_boxes = image_result.boxes.xyxyn.cpu().numpy()
        elif box_format == 'xywhn':
            image_boxes = image_result.boxes.xyxyn.cpu().numpy()
            image_boxes = [[float(x1), float(y1), float(np.abs(x1-x2)), float(np.abs(y1-y2))] for x1,y1,x2,y2 in image_boxes]
              
        image_scores = image_result.boxes.conf.cpu().numpy()
        image_cls = image_result.boxes.cls.cpu().numpy()
        
        for j in range(len(image_boxes)):
            (x1, y1, x2, y2), score, c = image_boxes[j], image_scores[j], image_cls[j]
            if score >= th:
                filtered_scores.append(score)
                filtered_cls.append(c)
                if box_format == 'xyxy':
                    filtered_crops.append(img.crop([x1,y1,x2,y2]))
                    filtered_boxes.append([int(x1),int(y1),int(x2),int(y2)])
                elif box_format == 'xyxyn':
                    filtered_crops.append(img.crop([x1*img_w, y1*img_h, x2*img_w, y2*img_h]))
                    filtered_boxes.append([float(x1),float(y1),float(x2),float(y2)])
                elif box_format == 'xywh':
                    filtered_crops.append(img.crop([x1,y1,x1+x2,y1+y2]))
                    filtered_boxes.append([int(x1),int(y1),int(x2),int(y2)])
                elif box_format == 'xywhn':
                    filtered_crops.append(img.crop([x1*img_w,y1*img_h,(x1+x2)*img_w,(y1+y2)*img_h]))
                    filtered_boxes.append([float(x1),float(y1),float(x2),float(y2)])
                    
        total_crops.append(filtered_crops)        
        total_boxes.append(filtered_boxes)
        total_scores.append(filtered_scores)
        total_cls.append(filtered_cls)
                

    return total_crops, total_boxes, total_scores, total_cls