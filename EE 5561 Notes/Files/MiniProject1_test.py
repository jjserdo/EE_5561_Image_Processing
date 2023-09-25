# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 21:11:31 2023

@author: justi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy

file = "fountain_girl.png"
raw = img.imread(file)   
image = np.asarray(raw)
print(image.dtype)
print(image.shape)
x1 = int(input("X - Coordinate of top left point: "))
y1 = int(input("Y - Coordinate of top left point: "))
x2 = int(input("X - Coordinate of bottom right point: "))
y2 = int(input("Y - Coordinate of bottom right point: "))
#target = np.full_like(image, 0)
#source = np.full_like(image, 0)