import cv2
import numpy as np
from time import time
import torch.backends.cudnn as cudnn
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
    total_boxes = []
    results = model.predict(imgs, stream=False)  
    
    for image_result in results:
        filtered_boxes = []
                
        if box_format == 'xyxy':
            image_boxes = image_result.boxes.xyxy.cpu().numpy()
        elif box_format == 'xywh':
            image_boxes = image_result.boxes.xywh.cpu().numpy()
        elif box_format == 'xyxyn':
            image_boxes = image_result.boxes.xyxyn.cpu().numpy()
        elif box_format == 'xywhn':
            image_boxes = image_result.boxes.xywhn.cpu().numpy()
              
        image_scores = image_result.boxes.conf.cpu().numpy()
        
        for j in range(len(image_boxes)):
            (x1, y1, x2, y2), score = image_boxes[j], image_scores[j]
            if score >= th:
                filtered_boxes.append([int(x1),int(y1),int(x2),int(y2), score])
        
        total_boxes.append(filtered_boxes)
                

    return total_boxes

cudnn.benchmark = True

    
# Initialize YOLO on GPU device.
detector = get_model('yolov8n.pt')
total = []
first_time =  time()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    
    start_time = time()
    # Detect face boxes on image.
    boxes = detect_faces(detector, [image], box_format='xyxy',th=0.4)[0]
    fps = 1/(time() - start_time)
    total.append(fps)

    # Draw detected faces on image.
    for left, top, right, bottom, score in boxes:
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
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