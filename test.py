from ultralytics import YOLO
import torch
import cv2
import argparse

def parse_variables():
    parser = argparse.ArgumentParser(description='Run Inference YOLOv8 for Face Detection')
    parser.add_argument('-w', '--weights', type=str, help='Path to trained weights',
                        default='runs/detect/train/weights/best.pt')
    parser.add_argument('-t', '--threshold', type=float, help='Score threshold',
                        default=0.5)
    
    parser.add_argument('-i', '--input', type=str, help='Sample input image path',
                        default='test_input.jpg')
    
    parser.add_argument('-o', '--output', type=str, help='Output image path',
                        default='test_output.jpg')

    
    args = parser.parse_args()
    variables = vars(args)
    return variables

def main():
    variables = parse_variables()
    
    # select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set threshold
    threshold = variables['threshold']

    # load a pretrained model (recommended for best training results)
    model = YOLO(variables['weights'])                
    model.to(device)
    
    # read image
    image = cv2.imread(variables['input'])
    
    # predict on an image
    results = model.predict(variables['input'])   

    

    # loop through results
    for result in results:
        for face in result['det']:
            x1, y1, x2, y2, score, c = face

            # if score is greater than threshold
            if score >= threshold:

                # draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (int(c == 0) * 255, int(c == 1) * 255, int(c ==2) * 255), 2)
    
    # plot image
    cv2.imwrite(variables['output'], image)
    

if __name__ == '__main__':
    main()
