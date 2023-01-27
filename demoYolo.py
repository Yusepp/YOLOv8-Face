import cv2
import numpy as np
from time import time
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from ultralytics import YOLO
import cv2
from detection import detect_faces
from PIL import Image

def get_model(weights):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(weights)                
    model.to(device)
    return model

cudnn.benchmark = True

    
# Initialize YOLO on GPU device.
detector = get_model('/home/yusepp/Desktop/DL/yolov8m.pt')
total = []
first_time =  time()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    
    start_time = time()
    # Detect face boxes on image.
    crops, boxes, scores, cls = detect_faces(detector, [Image.fromarray(image)], box_format='xywh',th=0.4)
    
    fps = 1/(time() - start_time)
    total.append(fps)

    # Draw detected faces on image.
    for (left, top, right, bottom), score in zip(boxes[0], scores[0]):
        cv2.rectangle(image, (int(left), int(top)), (int(left+right), int(top+bottom)), (255, 0, 0), 2)
        cv2.putText(image, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(image, f'Avg. FPS: {np.mean(total):.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(image, f'Max. FPS: {max(total):.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(image, f'Min. FPS: {min(total):.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(image, f"Face {score:.2f}",(int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    
    cv2.putText(image, f'{time()-first_time:.2f}s', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    cv2.imshow('YOLO DEMO', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    
cap.release()

# Nano = 55.4 fps ! 169 fps
# Medium = 27.5 fps ! 117 fps
#==================== Results ====================
# Easy   Val AP: 0.9759045555094775
# Medium Val AP: 0.9483292864330108
# Hard   Val AP: 0.7694889580283849
#=================================================