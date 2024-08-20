import os
import xml.etree.ElementTree as ET
import argparse
import random
import shutil

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(input_path, output_path, image_id, classes):
    in_file = open(os.path.join(input_path, f"{image_id}.xml"))
    out_file = open(os.path.join(output_path, f"{image_id}.txt"), "w")
    
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert_bbox((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def process_data(input_dir, output_dir, classes):
    images_dir = os.path.join(input_dir, 'images')
    annotations_dir = os.path.join(input_dir, 'annotations')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)

    split_index = int(0.8 * len(image_files))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    for img_file in train_files:
        shutil.copy(os.path.join(images_dir, img_file), os.path.join(train_dir, img_file))
        convert_annotation(annotations_dir, train_dir, os.path.splitext(img_file)[0], classes)

    for img_file in val_files:
        shutil.copy(os.path.join(images_dir, img_file), os.path.join(val_dir, img_file))
        convert_annotation(annotations_dir, val_dir, os.path.splitext(img_file)[0], classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Pascal VOC annotations to YOLO format and split data')
    parser.add_argument('input_dir', type=str, help='Path to input directory containing Pascal VOC annotations and images')
    parser.add_argument('output_dir', type=str, help='Path to output directory for processed data')
    args = parser.parse_args()

    classes = ["person", "hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]

    process_data(args.input_dir, args.output_dir, classes)

    print(f"Data processed and saved to {args.output_dir}")