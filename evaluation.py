import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import os
from PIL import Image
import torch
from tqdm import tqdm
from detection import get_model, detect_faces
import numpy as np
import argparse
from ensemble_boxes import *


parser = argparse.ArgumentParser(description='Run Inference YOLOv8 for Face Detection')
parser.add_argument('-e', '--eval', type=str, help='Path to WIDER_val/images',
                    default=os.path.join('WIDER_val','images'))

parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                        default='yolov8m_200e.pt')

parser.add_argument('-p', '--pred', type=str, help='Path to create evaluation .txt s',
                    default='WIDER_pred')

parser.add_argument('-t', '--threshold', type=float, help='Score threshold',
                    default=0.0001)

parser.add_argument('--multi-scale', action='store_true', help='Multi scale testing', default=False)

parser.add_argument('--vote', type=str, choices=['fusion','nms'], default='fusion', help='Bounding Box vote method')

args = parser.parse_args()
variables = vars(args)


model = get_model(variables['weights'])

eval_root = variables['eval']
prediction_root = variables['pred']
eval_list = os.listdir(eval_root)

#TEST_SCALES = [200, 500, 800, 1100, 1400, 1700, 2000, 2300]
TEST_SCALES = [500, 800, 1100, 1400, 1700]

if not os.path.exists(prediction_root):
    os.mkdir(prediction_root)

if variables['multi_scale']:
    print('Testing on multi-scale:', TEST_SCALES)
else:
    print('Testing on single-scale:')

for directory in tqdm(eval_list):
    current_pred_dir = os.path.join(prediction_root, directory)
    if not os.path.exists(current_pred_dir):
        os.mkdir(current_pred_dir)
    
    current_eval_dir = os.path.join(eval_root, directory)
    images = os.listdir(current_eval_dir)
    
    for image in images:
        image_array = Image.open(os.path.join(current_eval_dir, image))
        original_h, original_w = np.array(image_array).shape[:2]
        if not variables['multi_scale']:
            _, boxes, scores, cls = detect_faces(model, box_format='xywh',imgs=[image_array], th=variables['threshold'])
            boxes, scores, cls = boxes[0], scores[0], cls[0]
        else:
            all_boxes, all_scores, all_cls = [], [], []
            _, boxes, scores, cls = detect_faces(model, box_format='xyxyn',imgs=[image_array], th=variables['threshold'])
            if len(boxes) > 0:
                all_boxes, all_scores, all_cls = [boxes[0]], [scores[0]], [cls[0]]
            for scale in TEST_SCALES:
                new_size = (int(original_h*(original_w/scale)), int(scale))
                _, boxes, scores, cls = detect_faces(model, box_format='xyxyn',imgs=[image_array.resize(new_size)], th=variables['threshold'])
                boxes, scores, cls = boxes[0], scores[0], cls[0]
                if len(boxes) > 0:
                    all_boxes.append(boxes) 
                    all_scores.append(scores)
                    all_cls.append(cls)
            
            try:
                if variables['vote'] == 'nms':
                    boxes, scores, cls = nms(all_boxes, all_scores, all_cls, weights=None, iou_thr=0.5)
                elif variables['vote'] == 'fusion':
                    boxes, scores, cls = weighted_boxes_fusion(all_boxes, all_scores, all_cls, weights=None, iou_thr=0.5, skip_box_thr=0.0001)
                    
                boxes = [[x1*original_w, y1*original_h, abs(x1-x2)*original_w, abs(y1-y2)*original_h] for x1,y1,x2,y2 in boxes]
            except:
                boxes, scores, cls = [], [], []
            
            
        
        submission_file = os.path.join(current_pred_dir, image).replace(".jpg", ".txt")
        with open(submission_file, 'w', encoding="utf-8") as fw:
            fw.write(image.replace('.txt', '')+'\n')
            fw.write(f'{len(boxes)}\n')
            count = 0
            for (x,y,w,h), score, c in zip(boxes, scores, cls):
                count += 1
                if count < len(boxes):
                    fw.write(f'{round(x)} {round(y)} {round(w)} {round(h)} {score:.3f}\n')
                else:
                    fw.write(f'{round(x)} {round(y)} {round(w)} {round(h)} {score:.3f}')
        

