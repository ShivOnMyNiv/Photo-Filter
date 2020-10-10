# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:06:00 2020

@author: derph
"""

# Import the needed packages
import cv2
import numpy as np
from scipy import signal as sg
from PIL import Image
import matplotlib.pyplot as plt

# Creating the rounding function
def myround(x, base=40):
    return base * round(x/base)

# Importing image then viewing it
image = cv2.imread("Shopping Cart.jpg")
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(image.shape)

for y in range(0, len(image)):
    for x in range(0, len(image[y])):
        for z in range(0, 3):
            image[y][x][z] =  myround(image[y][x][z])
cv2.imshow("Remade Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()