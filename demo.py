from ultralytics import YOLO
import torch
import cv2
import argparse

def parse_variables():
    parser = argparse.ArgumentParser(description='Run Inference YOLOv8 for Face Detection')
    parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                        default='runs/detect/train/weights/best.pt')
    
    args = parser.parse_args()
    variables = vars(args)
    return variables

def main():
    variables = parse_variables()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = YOLO(variables['weights'])                
    model.to(device)

    results = model.predict('0', verbose=True)  
    

if __name__ == '__main__':
    main()
