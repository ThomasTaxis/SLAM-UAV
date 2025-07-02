import cv2
import numpy as np
import os



image_directory = r'C:\Users\ttaxi\Thesis\Calibration\2011_09_26\2011_09_26_drive_0119_extract\image_00\data'

image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

images = []
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(image_directory, image_file)
    
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    
   
   # Append keypoints and descriptors to the lists
  
    # Append the image to the list
    images.append(image)