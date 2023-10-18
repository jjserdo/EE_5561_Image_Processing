# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:27:41 2023

@author: justi
"""
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d

image = cv2.imread("images/raw/spider.png")
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
cv2.imwrite("images/spider.png", final_image)
"""
"""
cv2.imshow('Cropped and Resized Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
# Create a black image (mask) to draw on
#image = np.ones((400, 400, 3), dtype=np.uint8) * 255
image = cv2.imread("images/fountain_girl.png")
imageMask = np.ones(image.shape, dtype=np.uint8)*255

# Initialize variables for drawing
drawing = False
start_x, start_y = -1, -1

# Function to handle mouse events
def draw_circle(event, x, y, flags, param):
    global drawing, start_x, start_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(image, (start_x, start_y), (x, y), (0, 0, 0), 5)
            cv2.line(imageMask, (start_x, start_y), (x, y), (0, 0, 0), 5)
            start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        #cv2.line(image, (start_x, start_y), (x, y), (0,0,0), 5)

# Create a window and set the mouse event callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_circle)

while True:
    cv2.imshow('Image', image)
    
    # Press 'c' to clear the mask
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
    # Press 'q' to quit
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

_, binary_image = cv2.threshold(imageMask, 128, 255, cv2.THRESH_BINARY)
binary_array = (binary_image / 255).astype(float)

fig, ax = plt.subplots(figsize=(6,6), dpi=150)
ax.set_title("Image Mask")
ax.imshow(binary_image)
fig.tight_layout()
ax.axis('off')

fig, ax = plt.subplots(figsize=(6,6), dpi=150)
ax.set_title("Image Mask")
ax.imshow(binary_array)
fig.tight_layout()
ax.axis('off')

# Now you have the mask image stored in the 'image' variable. You can use it to erase from another image.
"""