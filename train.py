import os
import gc
import argparse

import shutil
from multiprocessing import freeze_support

import torch
from ultralytics import YOLO

def parse_variables():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Face Detection')
    parser.add_argument('-b', '--batch', type=int, help='Batch Size', default=8)
    parser.add_argument('-e', '--epochs', type=int, help='Number of Epochs', default=100)
    parser.add_argument('-w', '--workers', type=int, help='Workers for datalaoder', default=2)
    parser.add_argument('--pretrained', action='store_true', help='Finetune')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Train from zero.')
    parser.set_defaults(pretrained=False)
    parser.add_argument('-i', '--imgsize', type=int, help='Image size', default=640)
    parser.add_argument('-m', '--model',
                        type=str,
                        choices=['yolov8n','yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        default='yolov8m',
                        help='YOLOv8 size model')
    
    args = parser.parse_args()
    variables = vars(args)
    return variables

def main():
    # Parse the command line arguments
    variables = parse_variables()
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else device
    
    # Clean old run
    if os.path.exists('runs'):
        shutil.rmtree('runs')              

    # Choose model size to train
    model = YOLO('{}.pt'.format(variables['model']))                
    model.to(device)
    print(device)
    
    # Train the model
    _ = model.train(data='datasets/wider.yaml', epochs = variables['epochs'],
                    batch = variables['batch'], single_cls=True, pretrained=variables['pretrained'],
                    imgsz=variables['imgsize'], workers=variables['workers'])  
    
    # Evaluate whole validation dataset
    _ = model.val(data='datasets/wider.yaml')                          

if __name__ == '__main__':
    freeze_support()
    main()

    


