import pathlib
import torch
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# Load the model
model = torch.hub.load('.', 'custom', path='best.pt', source='local')

# Image path
img = r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\test\images\3.jpg'
conf_threshold = 0.5
# Inference
results = model(img)

# Show results
results.show()

# Extract bounding box coordinates
bounding_boxes = results.xyxy[0]  # First image in batch

# Iterate over bounding boxes and print them
for bbox in bounding_boxes:
    x1, y1, x2, y2, conf, cls = bbox
    print(f'Bounding Box: [{x1}, {y1}, {x2}, {y2}], Confidence: {conf}, Class: {cls}')