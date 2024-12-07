# YOLOv8-Image-Object-Dataset Integration
BY TOMMY LEE & U JIN SEAH

YOLOv8-MultiObjectDetection leverages YOLOv8 for multi-class object detection, and provides a workflow for combining multiple disparate datasets. It includes data preprocessing, class remapping, model training, and ONNX export for deployment, enabling robust and accurate detection for real-time vision applications.

Features
Dataset Integration: Combines multiple datasets with varying class mappings into a unified training dataset.
YOLOv8 Training: Trains a YOLOv8 model on combined datasets for multi-class object detection.
Class Remapping: Automatically remaps classes from different datasets into a unified structure.
Model Export: Supports exporting the trained model in ONNX format for deployment on various devices.
Customizable: Easily extendable for additional datasets or classes.

Getting Started
Prerequisites
Python 3.8 or later
Google Colab (recommended for training)
Installed libraries:
bash
Copy code
pip install ultralytics roboflow matplotlib opencv-python pyyaml
Dataset Preparation
Download Datasets:

Use Roboflow API to download datasets for various objects (wallets, keys, watches, etc.).
Datasets are remapped into a consistent class structure.
Dataset Directory Structure:

javascript
Copy code
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
Class Remapping:

Automatically remaps class indices to a unified format across all datasets.
Training the Model
Load the YOLOv8 model:

python
Copy code
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
Train the model:

python
Copy code
model.train(data='/content/final_dataset/data.yaml', epochs=10, imgsz=640, batch=16)
Inference
Use the trained model to make predictions:

python
Copy code
results = model.predict(source='/path/to/image.jpg', conf=0.3, save=True)
View results:

Processed images with bounding boxes are saved automatically.
Exporting the Model
Export the trained YOLOv8 model to ONNX format for deployment:

python
Copy code
model.export(format="onnx")
File Structure
data.yaml: Configuration file for the YOLOv8 model, specifying datasets and classes.
final_dataset/: Directory containing the unified training dataset.
notebook.ipynb: Google Colab notebook implementing the pipeline.
Results
Training Metrics: Accuracy, precision, recall, and loss are tracked during training.
Inference Output: Bounding boxes with class labels for multiple objects in a single image.
License
This project is licensed under the MIT License.

Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request.

Acknowledgments
Ultralytics YOLOv8
Roboflow



