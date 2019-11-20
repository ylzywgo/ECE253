#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import sobel


img = cv2.imread("geisel.jpg", 0)
img_pad = np.pad(img, (2, 2), mode = 'reflect')
# smoothing image
k = np.array([[2,4,5,4,2],
              [4,9,12,9,4],
              [5,12,15,12,5],
              [4,9,12,9,4],
              [2,4,5,4,2]]) * 1/159
img_smooth = np.zeros(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_smooth[i][j] = np.sum(img_pad[i : i + 5, j : j + 5] * k)

# gradient image
kx = np.array([[-1,0,1],
               [-2,0,2],
               [-1,0,1]])
ky = np.array([[-1,-2,-1],
               [0,0,0],
               [1,2,1]])
img_s_pad = np.pad(img_smooth, (1, 1), mode = 'reflect')
img_x = np.zeros(img.shape)
img_y = np.zeros(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_x[i][j] = np.sum(img_s_pad[i : i + 3, j : j + 3] * kx)
        img_y[i][j] = np.sum(img_s_pad[i : i + 3, j : j + 3] * ky)

gradient = np.sqrt(np.square(img_x) + np.square(img_y))
gradient *= 255 / gradient.max()
plt.title("original gradient magnitude image")
plt.imshow(gradient, cmap = 'gray')
plt.savefig("gradient_image.jpg")
plt.show()

angle = np.zeros(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (img_x[i][j] == 0):
            angle[i][j] = np.pi / 2
        else:
            angle[i][j] = np.arctan(img_y[i][j] / img_x[i][j])

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N))
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j+1]
                    r = img[i-1, j-1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j+1]
                    r = img[i+1, j-1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

img_nms = non_max_suppression(gradient, angle)
plt.title("the image after NMS")
plt.imshow(img_nms, cmap = 'gray')
plt.savefig("aft_NMS.jpg")
plt.show()

img_threshold = img_nms.copy()
img_threshold[img_threshold >= 130] = 255
img_threshold[img_threshold < 120] = 0
plt.title("the image after thresholding")
plt.imshow(img_threshold, cmap = 'gray')
plt.savefig("aft_threshold.jpg")
plt.show()

