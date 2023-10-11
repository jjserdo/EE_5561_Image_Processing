# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:14:56 2023

@author: justi
"""
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
# %% Original Image
image = cv2.imread("images/bw_fountain_girl.png")   
print(image.dtype)
print(image.shape)
image = np.copy(image[:,:,0])
fig, ax = plt.subplots()
ax.set_title("Original Image")
ax.imshow(image)

# %% Input Target 
targetType = [495,870,199,486]
x1, x2, y1, y2 = targetType
target = np.ones(image.shape, dtype=np.uint8)
print(target.dtype)
print(target.shape)
target[x1:x2, y1:y2] = 0
fig, ax = plt.subplots()
ax.set_title("Target Mask")
ax.imshow(target, cmap="gray")

# %% Viewing the thingy magic
def wowzers(image,target):
    new = np.zeros(image.shape, dtype=np.uint8)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            new[row,col] = image[row,col] * target[row,col] 
    plt.imshow(new)
    
new = image * target
#plt.imshow(new)

# %% Getting the contours of the mask
#kernel = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
kernely = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
kernel = kernelx + kernely
heyhey = np.abs(convolve2d(target, kernel, mode='same', boundary='wrap'))*255
heyhey = heyhey.astype(int)
print(heyhey.shape)
print(heyhey.dtype)
coordinates = []
for row in range(image.shape[0]):
    for col in range(image.shape[1]):
        if heyhey[row,col] != 0:
            coordinates.append([row,col])
fig, ax = plt.subplots()
ax.imshow(heyhey, cmap="gray")
ax.set_title("cmap")
    
# %% Contours of edges
xs = np.zeros(len(coordinates))
ys = np.zeros(len(coordinates))
for i in range(len(coordinates)):
    xs[i] = coordinates[i][0]
    ys[i] = coordinates[i][1]
print(len(coordinates))
fig, ax = plt.subplots()
ax.scatter(ys, xs)
ax.set_title("contours of edges")

# %% Data Term
a, b = cv2.spatialGradient(image)
h = a**2+b**2
fig, ax = plt.subplots()
ax.imshow(h, cmap="gray")
ax.set_title("data term")

# %% Getting the baby probabilities
n = 9
n2 = int(0.5*(n-1))
print(n2)
probTot = np.copy(target).astype(float)
probs = np.zeros(len(coordinates))

for coords in range(len(coordinates)):
    c, d = coordinates[coords]
    probs[coords] = np.sum(probTot[c-n2:c+n2,d-n2:d+n2])/(n**2)

# %% Getting the max probabilities
#boi = np.sort(probs.ravel())[::-1][:1]
boi = np.max(probs)
print(boi)
#sad = np.unravel_index(np.argsort(probs.ravel())[::-1][:1], image.shape)
sad = np.where(probs == boi)[0]
print(sad)
xss = np.zeros(len(sad))
yss = np.zeros(len(sad))
for i in range(len(sad)):
    xss[i], yss[i] = coordinates[sad[i]]
fig, ax = plt.subplots()
ax.scatter(yss, xss)
ax.set_title("max probabilities")

# %% multiplying the data term with the probability term and get max prio
hnorm = h/np.max(h)
priorities = np.zeros(len(coordinates))
for coords in range(len(coordinates)):
    c, d = coordinates[coords]
    priorities[coords] = probs[coords] * hnorm[c][d]
    
sadder = np.where(priorities == np.max(priorities))[0]
xsss = np.zeros(len(sadder))
ysss = np.zeros(len(sadder))
for i in range(len(sadder)):
    xsss[i], ysss[i] = coordinates[sadder[i]]
fig, ax = plt.subplots()
ax.scatter(ysss, xsss)
ax.set_title("max priorities")




    