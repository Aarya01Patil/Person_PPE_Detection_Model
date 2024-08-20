import os
import cv2
import argparse
import random
import xml.etree.ElementTree as ET
from ultralytics import YOLO

def pascalvoc_to_yolo_label(xml_file, img_width, img_height):
    yolo_labels = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        x1 = int(bndbox.find('xmin').text)
        y1 = int(bndbox.find('ymin').text)
        x2 = int(bndbox.find('xmax').text)
        y2 = int(bndbox.find('ymax').text)
        
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        class_id = class_names.index(class_name)  
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_labels

def crop_and_save_persons(images_dir, annotations_dir, output_dir, person_model, split_ratio=0.8):
    person_detector = YOLO(person_model)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    global class_names
    class_names = ["person", "hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.xml'
        label_path = os.path.join(annotations_dir, label_file)
        
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape
        
        if not os.path.exists(label_path):
            continue  

        yolo_labels = pascalvoc_to_yolo_label(label_path, img_width, img_height)
        
        results = person_detector(img)[0]
        
        for i, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            person_crop = img[y1:y2, x1:x2]
            
            if random.random() < split_ratio:
                save_dir = train_dir
            else:
                save_dir = val_dir
            
            crop_filename = f"{os.path.splitext(img_file)[0]}_person{i}.jpg"
            cv2.imwrite(os.path.join(save_dir, crop_filename), person_crop)


            yolo_label_path = os.path.join(save_dir, f"{os.path.splitext(crop_filename)[0]}.txt")
            with open(yolo_label_path, 'w') as f:
                for label in yolo_labels:
                    f.write(label + '\n')

    print(f"Cropped images and labels saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop persons from images and split for PPE training')
    parser.add_argument('--images_dir', type=str, help='Path to directory containing images')
    parser.add_argument('--annotations_dir', type=str, help='Path to directory containing Pascal VOC annotations')
    parser.add_argument('--output_dir', type=str, help='Path to output directory for cropped person images')
    parser.add_argument('--person_model', type=str, help='Path to person detection model weights')
    args = parser.parse_args()

    crop_and_save_persons(args.images_dir, args.annotations_dir, args.output_dir, args.person_model)
