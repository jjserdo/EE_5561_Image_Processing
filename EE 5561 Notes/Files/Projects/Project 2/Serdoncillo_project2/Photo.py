# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:27:41 2023

@author: justi
"""
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d

image = cv2.imread("images/cameraman.png")
print(image.dtype)
print(image.shape) 

fig, ax = plt.subplots(figsize=(6,6), dpi=150)
ax.set_title("Original Image")
ax.imshow(image)
fig.tight_layout()
ax.axis('off')

# Get the dimensions of the original image
height, width = image.shape[:2]

# Determine the size of the square (the smaller dimension)
size = min(width, height)

# Calculate the coordinates for cropping the square
start_x = (width - size) // 2
start_y = (height - size) // 2
end_x = start_x + size
end_y = start_y + size

# Crop the square
cropped_image = image[start_y:end_y, start_x:end_x]

final_image = cv2.resize(cropped_image, (200, 200))
cv2.imwrite("images/cameraman0.png", final_image)
