from ultralytics import YOLO
import torch
import cv2
import argparse

def parse_variables():
    parser = argparse.ArgumentParser(description='Run Inference YOLOv8 for Face Detection')
    parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                        default='yolov8m_200e.pt')
    
    args = parser.parse_args()
    variables = vars(args)
    return variables

def main():
    # Parse the command line arguments
    variables = parse_variables()
    
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else device
    

    # Load the pretrained model
    model = YOLO(variables['weights'])                
    model.to(device)

    # Run inference on the webcam
    results = model.predict('0', verbose=True)  
    

if __name__ == '__main__':
    main()
