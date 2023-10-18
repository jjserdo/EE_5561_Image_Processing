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
#image = cv2.imread("images/bw_fountain_girl.png")   
image = cv2.imread("images/blackWhite_paper.png")   
print(image.dtype)
print(image.shape)
image = np.copy(image[:,:,0])
fig, ax = plt.subplots()
ax.set_title("Original Image")
ax.imshow(image, cmap="gray")


# %% Input Target 
#targetType = [495,870,199,486]
targetType = [100,200,200,300]
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
    fig, ax = plt.subplots()
    ax.set_title("Magick Image")
    ax.imshow(new, cmap="gray")
    
wowzers(image, target)

# %% Getting the contours of the mask
#kernel = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
kernely = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
kernel = kernelx + kernely
heyhey = (np.abs(convolve2d(target, kernel, mode='same', boundary='wrap'))*255).astype(int)
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
    
# %% Contours of edges, getting all of the x,y coordinates
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
#print(n2)
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

# choosing a random value

# %% Getting the Source Patch

# Creating the source patch
new = np.zeros(image.shape, dtype=np.uint8)
for row in range(image.shape[0]):
    for col in range(image.shape[1]):
        new[row,col] = image[row,col] * target[row,col] 
    
# Convert to a 3d array for colorspace and convert
new3d = np.stack([new] * 3, axis = 2)
aa = np.random.randint(0, len(sadder))
patchP = coordinates[sadder[aa]]
a = patchP[0]
b = patchP[1]
compare = cv2.cvtColor(new3d[a-n2:a+n2+1,b-n2:b+n2+1], cv2.COLOR_BGR2LAB)

# get list of sources
allSource = [(i, j) for i in range(image.shape[0]) for j in range(image.shape[1]) if 
             np.any(target[i-n2:i+n2+1,j-n2:j+n2+1]==0) == False
             and i-n2 > 0 and i+n2 < image.shape[0] 
             and j-n2 > 0 and j+n2 < image.shape[1] ]

# get the colorspace stuff
ssd = np.zeros(len(allSource))
for i in range(len(allSource)):
    a = allSource[i][0]
    b = allSource[i][1]
   # if a-n2 < 0 or a+n2+1 > image.shape[0] or b-n2 < 0 or b+n2+1 > image.shape[1]:
    #    ssd[i] = np.nan 
    #else:
    compareWith = cv2.cvtColor(new3d[a-n2:a+n2+1,b-n2:b+n2+1], cv2.COLOR_BGR2LAB)
    ssd[i] = np.sum((compareWith-compare)**2)
    
saddest = np.where(ssd == np.nanmin(ssd))[0]
xssss = np.zeros(len(saddest))
yssss = np.zeros(len(saddest))
for i in range(len(saddest)):
    xssss[i], yssss[i] = allSource[saddest[i]]
fig, ax = plt.subplots()
ax.scatter(yssss, xssss, c='red')
ax.set_title("best source patch")

fig, ax = plt.subplots()
ax.scatter(ysss, xsss, c='b')
ax.scatter(yssss, xssss, c='r')
ax.imshow(new)

# choosing a random value from the max priorities and also from the max source patch
bb = np.random.randint(0, len(saddest))
patchQ = allSource[saddest[bb]]

# %% Patching up

evolve = np.copy(new)
a = patchP[0]
b = patchP[1]
c = patchQ[0]
d = patchQ[1]

# Update Image
evolve[a-n2:a+n2+1,b-n2:b+n2+1] =  evolve[c-n2:c+n2+1,d-n2:d+n2+1]
# Update Probabilities
probTot[a-n2:a+n2+1,b-n2:b+n2+1] = boi
# Update 
targetBool = np.copy(target)
targetBool[a-n2:a+n2+1,b-n2:b+n2+1] = 1

fig, ax = plt.subplots()
ax.imshow(evolve, cmap="gray")



    