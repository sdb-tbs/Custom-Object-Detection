# Custom Object Detection with YOLOv5

Process of training a YOLOv5 model on your own dataset for custom object detection.

## Prerequisites

- Python 3.8 or later
- PyTorch 1.7 or later
- Linux or macOS (Windows is supported via WSL)
- A CUDA-compatible GPU is highly recommended for faster training.

## Setup

1. **Clone the YOLOv5 Repository:**

   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
## Preparing Data
3. **Dataset Structure:**
Organize your dataset in the following structure:
   ```bash  
   dataset
      images
       train
       val
      labels
       train
       val
4. **Annotation Format**
   ```bash 
    <class> <x_center> <y_center> <width> <height>
Values are normalized from 0 to 1 relative to the image width and height.

## Configuring Your Training
5. **Edit the YAML File:**

Create or edit a .yaml file to define the number of classes and the path to your dataset,
