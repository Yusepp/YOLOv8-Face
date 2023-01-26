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

parser.add_argument('--multi-scale', action='store_true', help='Multi scale testing', default=False)

args = parser.parse_args()
variables = vars(args)


model = get_model(variables['weights'])

eval_root = variables['eval']
prediction_root = variables['pred']
eval_list = os.listdir(eval_root)


TEST_SCALES = [500, 800, 1100, 1400, 1700]

if not os.path.exists(prediction_root):
    os.mkdir(prediction_root)

print(variables)
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
    
    for image in tqdm(images):
        image_array = Image.open(os.path.join(current_eval_dir, image))
        original_w, original_h = np.array(image_array).shape[:2]
        if not variables['multi_scale']:
            _, boxes, scores, cls = detect_faces(model, box_format='xywh',imgs=[image_array], th=variables['threshold'])
            boxes, scores, cls = boxes[0], scores[0], cls[0]
        else:
            all_boxes, all_scores, all_cls = [], [], []
  
            for scale in TEST_SCALES:
                new_size = (int(scale), int(original_h*(original_w/scale)))
                _, boxes, scores, cls = detect_faces(model, box_format='xywhn',imgs=[image_array.resize(new_size)], th=variables['threshold'])
                boxes, scores, cls = boxes[0], scores[0], cls[0]
                all_boxes.append(boxes) 
                all_scores.append(scores)
                all_cls.append(cls)
            
            max_detections = max([len(r) for r in all_boxes])
            boxes, scores, cls = [], [], []
            for i in range(max_detections):
                final_box = np.mean([r[i] for r in all_boxes if i <= len(r) - 1], axis=0)
                final_score = np.mean([r[i] for r in all_scores if i <= len(r) - 1], axis=0)
                final_cls = np.mean([r[i] for r in all_cls if i <= len(r) - 1], axis=0)
                boxes.append([final_box[0]*original_w, final_box[1]*original_h, final_box[2]*original_w, final_box[3]*original_h])
                scores.append(final_score)
                cls.append(0 if int(final_cls) == 0 else 1)
            
        
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
        

