#!/usr/bin/env python
# coding: utf-8

# In[94]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[95]:


# def AHE(img, win_size):
win_size = 33
img = plt.imread("data/beach.png")
plt.imshow(img, cmap = 'gray')
plt.savefig("p1_original_image.jpg")
# plt.show()


# In[96]:


def AHE(img, win_size):
    height = img.shape[1]
    width = img.shape[0]
    
    pad_w = int(win_size / 2)
    pad_h = win_size - pad_w - 1
    img_pad = np.pad(img, (pad_w, pad_h), mode = 'symmetric')
    
    output = np.zeros((width, height))
    for x in range(0, width):
        for y in range(0, height):
            rank = 0
            pixel = img_pad[x + pad_w][y + pad_h]
            context_reg = img_pad[x : x + win_size, y : y + win_size]
            rank = np.sum(context_reg < pixel)
            output[x][y] = int(rank * 255 / win_size ** 2)
            
    return output


# In[100]:


win_sizes = [33, 65, 129]
i = 1

for win_size in win_sizes:
    output = AHE(img, win_size)

    plt.imshow(output, cmap = 'gray')
    plt.savefig("q1_AHE_%s.jpg"%i)
    i += 1
    plt.show()


# In[98]:


img = cv2.imread("data/beach.png",0)
equ = cv2.equalizeHist(img)
print(equ)
plt.imshow(equ, cmap='gray')
plt.savefig("p1_HE.jpg")


# 1.How does the original image qualitatively compare to the images after AHE and HE respectively?
# 
# The image after AHE and HE shows improved contrast.
# 
# 2.Which strategy (AHE or HE) works best for beach.png and why? Is this true for any image in general?
# 
# AHE shows better-improved contrast. As the histograms the adaptive method computes correspond to a distinct section of the image, and is used to redistribute the lightness values of the image. Therefore it will show better performance on improving the local contrast and enhancing the definitions of edges in each region of an image. 
# 
# However it is not true for any image in general as AHE has a tendency to overamplify noise in relatively homogeneous regions of an image. This drawback can also be shown in the AHE result of our image.
