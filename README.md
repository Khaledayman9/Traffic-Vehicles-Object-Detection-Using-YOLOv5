# Traffic-Vehicles-Object-Detection-Using-YOLOv5
An application designed for detecting and visualizing traffic-related objects in both images and videos. Leveraging the power of YOLOv5, this application is capable of identifying various traffic elements such as cars, number plates, and more, making it ideal for traffic monitoring, analysis, and related tasks.


# Dataset
The dataset[^1] contains labeled images of transport vehicles and number plates using LabelImg in YOLOv5 format. This dataset contains over 700 training samples, 185 validation samples, and over 200 test samples including videos. These images are provided along with corresponding bounding box annotations for training and evaluation. The images were labeled under 7 classes – Car, Number Plate, Blur Number Plate, Two Wheeler, Auto, Bus, and Truck in YOLOv5 format.

[^1]: [Traffic vehicles Object Detection](https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection)


# Aim
The aim of this application is to provide a comprehensive tool for detecting and visualizing traffic-related objects in both images and videos. By leveraging advanced object detection techniques, this application aims to enhance traffic monitoring, analysis, and management. It is designed to be adaptable to various datasets and use cases, making it a valuable resource for researchers, developers, and practitioners in the field of traffic and vehicle analysis.

# Objectives
1. Accurate Object Detection: Implement and fine-tune a state-of-the-art YOLOv5 object detection model to accurately identify and classify traffic-related objects, including cars, number plates, and other relevant categories.
2. Real-time Visualization: Develop functionalities to visualize detected objects in images and videos with bounding boxes, class labels, and confidence scores, ensuring clear and interpretable results.
3. Flexible Input Handling: Support multiple input formats, including various video formats (MP4, AVI, MOV) and image files, to accommodate diverse datasets and use cases.
4. Interactive Result Presentation: Utilize Matplotlib and other visualization tools to provide interactive plots and displays of detection results, including options to export frames as GIFs for easier sharing and review.
6. Customizability and Adaptability: Allow for easy adaptation of the model to different datasets and object categories by providing straightforward mechanisms for training with custom data and adjusting detection parameters.
7. Batch Processing Capabilities: Implement functionalities for batch processing of images and videos to facilitate large-scale analysis and ensure efficient handling of extensive datasets.

# Steps
## Data Preparation:
- Define Paths: Set paths for training, validation, and test images and labels in both the source dataset and the YOLOv5 format directories.
- Create YOLO Directories: Ensure that necessary directories for YOLO format data are created.
- Copy Images and Labels: Copy images and labels from the source directories to the YOLO format directories.
- Create data.yaml: Define dataset paths and class names in a data.yaml file used by YOLOv5 for training.
```python
data = {
    'train': os.path.join(yolo_data_path, 'images/train'),
    'val': os.path.join(yolo_data_path, 'images/val'),
    'test': os.path.join(yolo_data_path, 'images/test'),
    'nc': 7, 
    'names': ['Car', 'Number Plate', 'Blur Number Plate', 'Two Wheeler', 'Auto', 'Bus', 'Truck']
}
```

## Data Training:
- Train YOLOv5 Model: Train the YOLOv5 model using the specified configuration, data, and hyperparameters.
  - Epochs: 100
  - Image Size: 640x640
  - Batch Size: 16
  - cfg: yolov5s.yaml (model configuration)
```bash
!cd yolov5 && python detect.py --weights /kaggle/working/yolov5/runs/train/yolov5_traffic_detection/weights/best.pt --img-size 640 --conf-thres 0.4 --source {os.path.join(yolo_data_path, 'images/test')} --save-txt --save-crop
```
## Model Evaluation and Detection: 
- Run Inference: Use the trained model to predict bounding boxes on test images and save the results.
  - img-size: 640 (input image size for inference)
  - conf-thres: 0.4 (confidence threshold for predictions)
```bash
!cd yolov5 && python detect.py --weights /kaggle/working/yolov5/runs/train/yolov5_traffic_detection/weights/best.pt --img-size 640 --conf-thres 0.4 --source {os.path.join(yolo_data_path, 'images/test')} --save-txt --save-crop
```
## Result Visualization:
- Display Predictions: Implement functions to visualize predictions on images and videos, showing bounding boxes with class labels and probabilities.
- Draw Bounding Boxes: Use matplotlib to draw bounding boxes and annotations on frames from video files.
- Process Video: Extract and visualize frames from video files, applying the same detection and visualization as for images.

## Model Export:
- Export Model: Export the trained YOLOv5 model to ONNX format for deployment and further use.
```bash
!python yolov5/export.py --weights /kaggle/working/yolov5/runs/train/yolov5_traffic_detection/weights/best.pt --img-size 640 --batch-size 1 --device cpu --include onnx
```
## Prediction on External Samples:
- Load and Predict: Load the exported model and predict bounding boxes on a new set of images.

