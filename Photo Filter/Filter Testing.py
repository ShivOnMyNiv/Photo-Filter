# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:13:32 2020

@author: derph
"""

# import the necessary packages
import cv2
import numpy as np
from scipy import signal as sg
from PIL import Image

cam = cv2.VideoCapture(0)

image = cv2.imread("Abhi Ganguly.jpg")

# Checking that image was loaded
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Assembling Kernels
kernel = {}
kernel["BB"] = np.ones((3, 3)) * 0.111
kernel["GB"] = GaussianBlur = (1/16) * np.array([[1,2,1],
                                                 [2,4,2],
                                                 [1,2,1]])
kernel["ED"] = np.array([[-1,-1,-1],
                          [-1,8,-1],
                          [-1,-1,-1]])
kernel["ED2"] = np.array([[0,-1,0],
                          [-1,4,-1],
                          [0,-1,0]])
kernel["SH"] = np.array([[0,-1,0],
                         [-1,5,-1],
                         [0,-1,0]])
print(kernel)

# Creating rounded photo
print("Making comic filter...")
def myround(x, base=40):
    return base * round(x/base)
comic = image.copy()
for y in range(0, len(image)):
    for x in range(0, len(image[y])):
        for z in range(0, 3):
            comic[y][x][z] =  myround(image[y][x][z])

# Testing kernels 
cv2.startWindowThread()
blur = cv2.GaussianBlur(image, (3, 3), 0)
sharp = cv2.addWeighted(image, 1.5, blur, -0.5, 1)
canny = cv2.Canny(image, 100, 300)
kernel1 = sg.convolve(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), kernel["ED"], mode="same")/255
kernel2 = sg.convolve(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), kernel["ED2"], mode="same")/255
cv2.imshow("original", image)
#cv2.imshow("blur", blur)
cv2.imshow("Sharp", sharp)
cv2.imshow("Canny", canny)
cv2.imshow("Kernel1", kernel1)
#cv2.imshow("Kernel2", kernel2)
cv2.imshow("Comic", comic)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
# Sharpening borders
border = []
for y in range(0, len(kernel1)):
    for x in range(0, len(kernel1[y])):
        if kernel1[y][x]*255 >= 50:
            image[y][x][0] = 0
            image[y][x][1] = 0
            image[y][x][2] = 255
            border.extend([(y, x),(y, x),(y, x)])
        else:
            for z in range(0, 3):
                image[y][x][z] = myround(image[y][x][z])
cv2.imshow("Border Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()             
"""
            
# Saving images
for y in range(0, len(kernel1)):
    for x in range(0, len(kernel1[y])):
        if kernel1[y][x] < 0:
            kernel1[y][x] = 0
print(kernel1.shape)
cv2.imshow("Outline", kernel1)
cv2.waitKey(0)
cv2.destroyAllWindows()


