#
# Company:    	    AIStorm 
# Engineer:	        Jungwirth Martin
# Copyright (c)     2019-2023 AIStorm Inc. All rights reserved.	   
# Project Name:     Cheetah Receiver 
# Create Date:      10.11.2023 
#
import glob
import os.path
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
import cv2
import torch
from yolov5 import YOLOv5
import numpy as np
from PIL import Image

class GunDetect():
  
  def __init__(self):
    
    pass

  def _preprocess_array_to_image(self, arr):
    """
    covert the type to uint8
    graytobgr
    """
    # Convert to uint8 without changing values
    if arr.dtype != np.uint8:
      arr = arr.astype(np.uint8)

    # Convert grayscale to RGB if necessary
    if len(arr.shape) == 2:  # Grayscale image
      arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return arr
  
  def run(self, img):
    original_img = img.copy()
    conf_thresh = 0.001


    # img = self._preprocess_array_to_image(img)
    # img = cv2.resize(converted, (160, 160))
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img = Image.fromarray(img)
    # print(img)

    overlay = []          # List of Bounding Boxes
    model = torch.hub.load('.', 'custom', path='yolov5m.pt', source='local')
    # model = YOLOv5('best.pt', device="cpu")

    # Inference
    # results = model.predict(img)
    results = model(img, size = 160)
    # print(results)
    bounding_boxes = results.xyxy[0]  # First image in batch
    conf = None

    for bbox in bounding_boxes:

      #xmin, ymin, xmax, ymax
      x1, y1, x2, y2, conf, cls = bbox
      if conf > conf_thresh:
        overlay.append((x1.item(), y1.item(), x2.item(), y2.item()))
        cv2.rectangle(original_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        print(f'Bounding Box: [{x1}, {y1}, {x2}, {y2}], Confidence: {conf}, Class: {cls}')

    return original_img, overlay
obj = GunDetect()
# img =cv2.imread(r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Rifle.v1i.yolov5pytorch\valid\images\19_jpeg.rf.fd848e430bdeb42f175baf4095a232ed.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def read_images(directory):
  image_files = []
  for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif'):
    image_files.extend(glob.glob(os.path.join(directory, ext)))
  return image_files
# img = np.load('your_np_file')
def save_bbx_info(file_path, image_shape, bboxes):
  h, w, _ = image_shape
  with open(file_path, 'w') as f:
      for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        # Convert bounding box to YOLO format
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        bbox_width = (x2 - x1) / w
        bbox_height = (y2 - y1) / h
        line = f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n"
        f.write(line)
ims = read_images(r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Nerf_Gun_video\1')
output_dir = r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Nerf_Gun_video\1_bbx'
for im in ims:
    img = cv2.imread(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_img, bboxes = obj.run(img)

    base_name = os.path.basename(im)
    output_image_path = os.path.join(output_dir, base_name)
    output_bbx_path = os.path.join(output_dir, os.path.splitext(base_name)[0] + '.txt')

    cv2.imwrite(output_image_path, processed_img)
    save_bbx_info(output_bbx_path, processed_img.shape, bboxes)