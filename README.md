# YOLOv8-Image-Object-Dataset Integration & Object Detection
BY TOMMY LEE & U JIN SEAH

## Overview

YOLOv8-MultiObjectDetection leverages YOLOv8 for multi-class object detection and provides a streamlined workflow for integrating multiple disparate datasets. This project enables robust, real-time vision applications by offering tools for data preprocessing, class remapping, model training, and ONNX export for deployment.

![GIF-Live-Demo-2](https://github.com/user-attachments/assets/9f40dc05-9e32-40e8-9692-3e7b73ebc8a8)


### **Why This Matters**
---
## Features

- **Dataset Integration**: Combines datasets with varying class mappings into a unified training dataset.
- **YOLOv8 Training**: Trains a YOLOv8 model for multi-class object detection.
- **Class Remapping**: Automatically remaps classes from different datasets into a unified structure.
- **Model Export**: Exports the trained model in ONNX format for deployment on edge devices or cloud platforms.
- **Customizable**: Easily extendable to incorporate additional datasets or classes.

---

## Table of Contents

1. Getting Started
2. Dataset Preparation
3. Training the Model
4. Inference
5. Exporting the Model
6. Results
7. File Structure
8. License
9. Contributions
10. Acknowledgements

---

## Getting Started

### Prerequisites

- **Python**: Version 3.8 or later.
- **Google Colab**: Recommended for training.
- Install required libraries:
    
    ```
   
   pip install ultralytics IPython.display roboflow matplotlib opencv-python pyyaml
    
    ```

---

## Dataset Preparation

### Download Datasets

Use the **Roboflow API** to download datasets for various objects (e.g., wallets, phones, watches). Class mappings are remapped into a consistent structure.

### Directory Structure

```
/dataset-name/
  /train/
    /images/
    /labels/
  /valid/
    /images/
    /labels/
  /test/
    /images/
    /labels/

```

---

## Training the Model

1. **Load YOLOv8 model**:
    
    ```
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    
    ```
    
2. **Train the model**:
    
    ```
    model.train(data='/content/final_dataset/data.yaml', epochs=10, imgsz=640, batch=16)
    
    ```
    

---

## Inference

Use the trained model to make predictions:

```
model = YOLO("/content/runs/detect/train/weights/best.pt")
image_path = "/content/test.jpg"
results = model.predict(source=image_path, conf=0.3, save=True)

```

Processed images with bounding boxes are saved automatically.

---

## Exporting the Model

Export the trained model for deployment:

```

model.export(format="onnx")

```

---

## Results

### Training Metrics

- **Accuracy**: 90%+
- **Precision/Recall**: High performance across multiple classes.

### Example Output

Processed images with bounding boxes:

![Screen Recording 2025-01-22 at 1 17 53 AM 00_00_31_05 Still014](https://github.com/user-attachments/assets/f88352c8-e4b7-4f2b-899f-51c4b0f20464)

---

## File Structure

- **data.yaml**: Configuration file specifying datasets and classes.
- **final_dataset/**: Unified training dataset.
- **notebook.ipynb**: Google Colab notebook implementing the pipeline.

---

## License

This project is licensed under the MIT License.

---

## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8)
- [Roboflow](https://roboflow.com/)
---
