import argparse
import os
import cv2
import torch
from ultralytics import YOLO

def convert_ppe_annotations_to_full_image(ppe_annotations, person_bbox):
    x1, y1, x2, y2 = person_bbox
    converted_annotations = []
    
    for ppe_bbox in ppe_annotations:
        px1, py1, px2, py2 = ppe_bbox
        fx1, fy1 = x1 + px1, y1 + py1
        fx2, fy2 = x1 + px2, y1 + py2
        converted_annotations.append((fx1, fy1, fx2, fy2))
    
    return converted_annotations

def draw_boxes(img, boxes, labels, scores):
    class_colors = {
        "person": (0, 0, 255),
        "hard-hat": (0, 255, 0),
        "gloves": (0, 255, 255),
        "mask": (255, 255, 0),
        "glasses": (255, 0, 255),
        "boots": (255, 165, 0),
        "vest": (255, 182, 193),  
        "ppe-suit": (148, 0, 211),
        "ear-protector": (0, 128, 128),
        "safety-harness": (128, 128, 128),
    }
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = class_colors.get(label, (255, 255, 255))  # Default to white if class not found
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label_text = f"{label}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)

def process_frame(frame, person_detector, ppe_detector):
    person_results = person_detector(frame)[0]
    
    person_boxes = []
    person_labels = []
    person_scores = []
    
    for person_box in person_results.boxes:
        x1, y1, x2, y2 = map(int, person_box.xyxy[0])
        score = person_box.conf.item()
        person_boxes.append((x1, y1, x2, y2))
        person_labels.append("person")
        person_scores.append(score)
        
        person_crop = frame[y1:y2, x1:x2]
        ppe_results = ppe_detector(person_crop)[0]
        
        ppe_annotations = []
        ppe_labels = []
        ppe_scores = []
        
        for ppe_box in ppe_results.boxes:
            # Extract bounding box coordinates
            px1, py1, px2, py2 = map(int, ppe_box.xyxy[0])
            # Extract the class ID and confidence
            ppe_class = ppe_box.cls.item()
            ppe_conf = ppe_box.conf.item()
            
            # Append to the list of annotations, labels, and scores
            ppe_annotations.append((px1, py1, px2, py2))
            ppe_labels.append(ppe_results.names[int(ppe_class)])
            ppe_scores.append(ppe_conf)
        
        # Convert PPE annotations to full image coordinates
        full_image_ppe_annotations = convert_ppe_annotations_to_full_image(
            ppe_annotations,
            (x1, y1, x2, y2)
        )
        
        # Draw the PPE bounding boxes and labels on the full image
        draw_boxes(frame, full_image_ppe_annotations, ppe_labels, ppe_scores)
    
    # Draw the person bounding boxes and labels on the full image
    draw_boxes(frame, person_boxes, person_labels, person_scores)
    
    return frame


def perform_inference(input_dir, output_dir, person_model, ppe_model):
    person_detector = YOLO(person_model)
    ppe_detector = YOLO(ppe_model)

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        file_name, file_ext = os.path.splitext(file)
        
        if file_ext.lower() in ['.jpg', '.jpeg', '.png']:
            image = cv2.imread(file_path)
            processed_image = process_frame(image, person_detector, ppe_detector)
            output_path = os.path.join(output_dir, file)
            cv2.imwrite(output_path, processed_image)
            
        elif file_ext.lower() == '.mp4':
            video = cv2.VideoCapture(file_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_path = os.path.join(output_dir, f"{file_name}_processed.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                processed_frame = process_frame(frame, person_detector, ppe_detector)
                out.write(processed_frame)
            
            video.release()
            out.release()
        
        else:
            print(f"Unsupported file format: {file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform inference on images and videos')
    parser.add_argument('--input_dir', type=str, help='Path to input directory containing images and videos')
    parser.add_argument('--output_dir', type=str, help='Path to output directory for processed images and videos')
    parser.add_argument('--person_model', type=str, help='Path to person detection model weights')
    parser.add_argument('--ppe_model', type=str, help='Path to PPE detection model weights')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    perform_inference(args.input_dir, args.output_dir, args.person_model, args.ppe_model)
