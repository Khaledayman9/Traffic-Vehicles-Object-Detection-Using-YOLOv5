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
!export WANDB_MODE=disabled && cd yolov5 && python train.py --img-size 640 --batch-size 16 --epochs 100 --data {data_yaml_path} --cfg yolov5s.yaml --weights '' --name yolov5_traffic_detection --cache
```
- Training Statistics (Epoch 99):
    - GPU Memory Usage: 3.08 GB
    - Box Loss: 0.03602
    - Object Loss: 0.06729
    - Class Loss: 0.008569
    - Instances per Batch: 12
    - Image Size: 640x640
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

# Results:
- After the training process, the model achieved the following results:
    - Precision (P): 0.74
    - Recall (R): 0.645
    - Mean Average Precision at IoU=0.5 (mAP50): 0.704
    - mAP50-95 (Average Precision across IoU thresholds from 0.5 to 0.95): 0.4
- Class-wise Metrics:
    - Car:
        - Precision (P): 0.81
        - Recall (R): 0.873
        - mAP50: 0.9
        - mAP50-95: 0.642
    - Number Plate:
        - Precision (P): 0.752
        - Recall (R): 0.776
        - mAP50: 0.778
        - mAP50-95: 0.323
    - Blur Number Plate:
        - Precision (P): 0.799
        - Recall (R): 0.47
        - mAP50: 0.584
        - mAP50-95: 0.28
    - Two Wheeler:
        - Precision (P): 0.791
        - Recall (R): 0.764
        - mAP50: 0.821
        - mAP50-95: 0.455
    - Auto:
        - Precision (P): 0.785
        - Recall (R): 0.5
        - mAP50: 0.569
        - mAP50-95: 0.261
    - Bus:
       - Precision (P): 0.691
       - Recall (R): 0.664
       - mAP50: 0.725
       - mAP50-95: 0.442
    - Truck:
        - Precision (P): 0.554
        - Recall (R): 0.468
        - mAP50: 0.551
        - mAP50-95: 0.397
- The final metrics indicate that the model performs well, especially in detecting Cars and Two Wheelers. The model's performance metrics show improvements in precision, recall, and mAP50 compared to earlier epochs. The mAP50-95 value provides a broader measure of performance across different IoU thresholds, reflecting overall robustness in object detection.

# Samples from Test Set
## Image Samples:

![1](https://github.com/user-attachments/assets/686adcb2-6e9a-4b2e-854f-bd18f729d709)

![2](https://github.com/user-attachments/assets/74dd8cce-684b-4974-b7bf-ef553618ea11)

![3](https://github.com/user-attachments/assets/cb4ef8ac-af48-4877-aa63-3ed78aca1413)

![4](https://github.com/user-attachments/assets/2a0bb46c-0684-43c0-b50f-3992aaee7b75)

![6](https://github.com/user-attachments/assets/919c89eb-64c3-4c5b-9712-3c21abbe8f71)

![5](https://github.com/user-attachments/assets/4f0cc0ab-8e2f-4a3a-a000-7c9484d92ca5)

![7](https://github.com/user-attachments/assets/e4ef0690-4089-4c27-b867-d5529163ed2e)

## Video Samples:
- Plotting the vehicle objects across all frames. Here are the last frames from two random video samples from the test set.

![33](https://github.com/user-attachments/assets/871eb2e0-a242-4f72-9db7-136bf8d4076b)

![22](https://github.com/user-attachments/assets/5b6ec928-773b-4b74-a51f-71b8fd445519)

# Conclusion
The training of your YOLOv5 model for traffic vehicle detection has yielded positive results, demonstrating both robustness and accuracy.
- **Overall Performance:** The model achieved a mean average precision (mAP50) of 0.704 on the validation set, indicating strong performance in detecting and localizing objects with an Intersection over Union (IoU) threshold of 0.5. This level of accuracy suggests that the model is effective in identifying objects in various scenarios and conditions.

- **Class-Specific Insights:**
        - Cars are detected with high precision (0.81) and recall (0.873), achieving a high mAP50 of 0.9, indicating excellent performance in identifying cars.
        - Two Wheelers and Number Plates also show strong performance with mAP50 values of 0.821 and 0.778, respectively, though the model's performance on Blur Number Plates and Auto is comparatively lower.
        - The detection of Trucks and Buses is less robust, as indicated by lower precision and recall values. This might suggest the need for further fine-tuning or additional training data for these classes.
- **Loss Metrics:** The low box loss (0.03602), object loss (0.06729), and class loss (0.008569) at the final epoch reflect the model's effectiveness in minimizing prediction errors, contributing to its overall accuracy.
- **Model Efficiency:** With a training time of just 0.547 hours and a model complexity of 15.8 GFLOPs, the YOLOv5s architecture provides a good balance between performance and computational efficiency, making it suitable for real-time applications.
In summary, the model demonstrates strong overall performance with particular strengths in detecting cars and two-wheelers. The lower performance on certain classes suggests areas for potential improvement, such as gathering more data or adjusting model parameters to enhance detection accuracy for those classes.

# Acknowledgments
We would like to acknowledge the YOLOv5[^2] repository for its invaluable contribution to the object detection training process. The YOLOv5 architecture and its associated tools provided the foundation and functionality necessary to develop and train our car detection model efficiently. We appreciate the efforts and open-source contributions of the YOLOv5 team, which have significantly enhanced the performance and capabilities of our project.

[^2]: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)

# Technologies
- Python
- Kaggle Notebook
