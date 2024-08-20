# Person & PPE Detection Using YOLOv8

## Project Overview

This repository provides a comprehensive object detection system leveraging YOLOv8 to identify individuals and personal protective equipment (PPE) in various environments. The system is designed to handle a dataset with annotations for PPE categories such as hard hats, gloves, masks, glasses, boots, vests, PPE suits, ear protectors, and safety harnesses. The project aims to train two distinct models: one for detecting persons in complete images and another for detecting PPE in cropped images of detected persons.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
- [Annotation Conversion](#annotation-conversion)
- [Model Training](#model-training)
- [Inference](#inference)
- [Evaluation Metrics](#evaluation-metrics)
- [Report](#report)
- [Demonstration Video](#demonstration-video)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aarya01Patil/Person_PPE_Detection_Model.git
cd Person_PPE_Detection_Model
```

### Step 2: Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare the Dataset

Download the dataset from the following [link](https://drive.google.com/file/d/1myGjrJZSWPT6LYOshF9gfikyXaTCBUWb/view?usp=sharing). Extract the contents of `Datasets.zip`, which includes the `images` and `annotations` directories along with `classes.txt`.

### Step 2: Convert Annotations

Convert annotations from PascalVOC format to YOLOv8 format using the command below:

```bash
python pascalVOC_to_yolo.py --input_dir path/to/inputdirectory --output_dir path/to/outputdirectory
```

### Step 3: Train the Models

#### Train Person Detection Model

To train the YOLOv8 model for detecting persons, execute:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data\processed\person\data.yaml epochs=100 imgsz=640 batch=16                                                                                                     
```

#### Train PPE Detection Model

To train the YOLOv8 model for PPE detection on cropped images, first ensure that you have implemented the logic to crop images based on detected persons. Then, run:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data\processed\ppe\data.yaml epochs=100 imgsz=640 batch=16                                                                                                    
```

### Step 4: Run Inference

Perform inference using both trained models with the following command:

```bash
python inference.py --input_dir path/to/images --output_dir path/to/output --person_model path/to/weights/person_detection/best.pt --ppe_model path/to/weights/ppe_detection/best.pt
```


## Report

A detailed report covering the methodologies, learning outcomes, and evaluation metrics is available in PDF format. It includes:

- The logic employed for model training.
- Challenges encountered and solutions implemented.

Access the [Project Report](https://docs.google.com/document/d/1KyKaH_2FiUJMdyso3dG3D6ZzquYdpyNPmsRKmVV8F9M/edit).

## Tutorial / Demonstration Video

A demonstration video of the project can be viewed [here](https://www.loom.com/share/1210bb6a538e4701a8a7cd3a5b9e54db?sid=8290368b-8bdd-40b4-9c7a-03f1d220f355).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

