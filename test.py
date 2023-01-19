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
    
    parser.add_argument('-i', '--input', nargs='+', help='Sample input image path',
                        default=['test_images/test_input.jpg','test_images/test_input_2.jpg','test_images/test_input_3.jpg'])

    parser.add_argument('-o', '--output', nargs='+', help='Sample output image path',
                        default=['test_images/test_output.jpg', 'test_images/test_output_2.jpg', 'test_images/test_output_3.jpg'],)


    
    args = parser.parse_args()
    variables = vars(args)
    return variables

def main():
    variables = parse_variables()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold = variables['threshold']

    model = YOLO(variables['weights'])                
    model.to(device)
    
    imgs = [cv2.imread(f) for f in variables['input']]
    results = model.predict(imgs, verbose=True)  

    for i, result in enumerate(results):
        image = imgs[i]
        bboxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()
        
        for j in range(len(bboxes)):
            (x1, y1, x2, y2), score, c = bboxes[j], scores[j], cls[j]

            if score >= threshold:

                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (int(c == 0) * 255, int(c == 1) * 255, int(c ==2) * 255), 2)

        cv2.imwrite(variables['output'][i], image)
    

if __name__ == '__main__':
    main()
