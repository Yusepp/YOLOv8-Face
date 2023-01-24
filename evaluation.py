import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import os
from PIL import Image
import torch
from tqdm import tqdm
from detection import get_model, detect_faces
import numpy as np
import argparse



parser = argparse.ArgumentParser(description='Run Inference YOLOv8 for Face Detection')
parser.add_argument('-e', '--eval', type=str, help='Path to WIDER_val/images',
                    default=os.path.join('WIDER_val','images'))

parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                        default='yolov8m_200e.pt')

parser.add_argument('-p', '--pred', type=str, help='Path to create evaluation .txt s',
                    default='WIDER_pred')

parser.add_argument('-t', '--threshold', type=float, help='Score threshold',
                    default=0.05)

args = parser.parse_args()
variables = vars(args)


model = get_model(variables['weights'])

eval_root = variables['eval']
prediction_root = variables['pred']
eval_list = os.listdir(eval_root)

if not os.path.exists(prediction_root):
    os.mkdir(prediction_root)


for directory in tqdm(eval_list):
    current_pred_dir = os.path.join(prediction_root, directory)
    if not os.path.exists(current_pred_dir):
        os.mkdir(current_pred_dir)
    
    current_eval_dir = os.path.join(eval_root, directory)
    images = os.listdir(current_eval_dir)
    
    for image in tqdm(images):
        image_array = Image.open(os.path.join(current_eval_dir, image))
        _, boxes, scores, cls = detect_faces(model, box_format='xywh',imgs=[image_array], th=variables['threshold'])
        boxes, scores, cls = boxes[0], scores[0], cls[0]
        
        submission_file = os.path.join(current_pred_dir, image).replace(".jpg", ".txt")
        with open(submission_file, 'w', encoding="utf-8") as fw:
            fw.write(image.replace('.txt', '')+'\n')
            fw.write(f'{len(boxes)}\n')
            count = 0
            for (x,y,w,h), score, c in zip(boxes, scores, cls):
                count += 1
                if count < len(boxes):
                    fw.write(f'{x} {y} {w} {h} {score:.3f}\n')
                else:
                    fw.write(f'{x} {y} {w} {h} {score:.3f}')
        

