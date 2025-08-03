import yaml
import json
import cv2
from ultralytics import YOLO
import argparse 

parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Pipeline")
parser.add_argument('--image', type=str, required=True, help="Path to the input image file.")
args = parser.parse_args()

def detect_objects(image_path, model_path, confidence_threshold):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print(f"Loading image from {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return []
    print("Running detection...")
    results = model(img)
    detections = []
    names = results[0].names
    for box in results[0].boxes:
        if box.conf[0] >= confidence_threshold:
            x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = names[class_id]
            detection_info = {
                'class_name': class_name,
                'confidence': round(confidence, 4),
                'bounding_box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            }
            detections.append(detection_info)
    print(f"Detection complete. Found {len(detections)} objects.")
    return detections

def save_output(detections, output_path):
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    config_path = 'config.yaml' 
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    input_image_path = args.image

    detected_objects = detect_objects(
        image_path=input_image_path,
        model_path=config.get('model_path'),
        confidence_threshold=config.get('confidence_threshold')
    )

    if detected_objects:
        save_output(detected_objects, config.get('output_path'))