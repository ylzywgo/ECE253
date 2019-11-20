#!/usr/bin/env python
# coding: utf-8

# # Problem 2

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import opening, closing, dilation, erosion
from skimage.morphology import disk, rectangle
import pandas
from scipy.ndimage import label
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ### Part(i)

# In[2]:


img = cv2.imread("data/circles_lines.jpg", 0)
img[img <= 128] = 0
img[img > 128] = 1


# In[3]:


img_new = opening(img, disk(4))
labeled_array, num_features = label(img_new)
print(num_features)


# In[4]:


f1 = plt.figure(figsize=(12,12))
f1_ax1 = f1.add_subplot(221)
f1_ax2 = f1.add_subplot(222)
f1_ax3 = f1.add_subplot(223)
f1_ax1.imshow(img, cmap='gray')
f1_ax1.title.set_text("original image")
f1_ax2.imshow(img_new, cmap='gray')
f1_ax2.title.set_text("image after opening")
img3 = f1_ax3.imshow(labeled_array, cmap='nipy_spectral')
f1_ax3.title.set_text("image after connected component labeling")
divider = make_axes_locatable(f1_ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(img3, cax, orientation='vertical')
plt.savefig("p2_circles_lines.jpg")
plt.show()

data = []
for i in range(num_features):
    position = np.where(labeled_array == i + 1)
    area = len(position[0])
    x_cen = round(sum(position[0]) / area, 2)
    y_cen = round(sum(position[1]) / area, 2)
    data.append([area, (round(x_cen,2), round(y_cen,2))])
    
title = ["area", "centroid"]
circle = list(np.arange(1, 31, 1))
print(pandas.DataFrame(data, circle, title))


# i) The original image, the image after opening, the image after connected component labeling (plot with colorbar), and a table with the desired values for each component of part(i) is shown above. 
# 
# ii) The structure used for the opening operation is disk and the size of it is 4.
# 
# iii) codes shown above

# ### Part(ii)

# In[5]:


img_l = cv2.imread("data/lines.jpg", 0)
img_l[img_l <= 128] = 0
img_l[img_l > 128] = 1


# In[6]:


img_nl = opening(img_l, rectangle(8,1))
labeled_array_l, num_features_l = label(img_nl)
print(num_features_l)


# In[7]:


f2 = plt.figure(figsize=(14,12))
f2_ax1 = f2.add_subplot(221)
f2_ax2 = f2.add_subplot(222)
f2_ax3 = f2.add_subplot(223)
f2_ax1.imshow(img_l, cmap='gray')
f2_ax1.title.set_text("original image")
f2_ax2.imshow(img_nl, cmap='gray')
f2_ax2.title.set_text("image after opening")
imgl_3 = f2_ax3.imshow(labeled_array_l, cmap='nipy_spectral')
f2_ax3.title.set_text("image after connected component labeling")
divider = make_axes_locatable(f2_ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(imgl_3, cax, orientation='vertical')
plt.savefig("p2_lines.jpg")
plt.show()

data_l = []
for i in range(num_features_l):
    position = np.where(labeled_array_l == i + 1)
    x_len = max(position[0]) - min(position[0])
    y_len = max(position[1]) - min(position[1])
    length = np.sqrt(x_len**2+y_len**2)
    x_cen = sum(position[0]) / len(position[0])
    y_cen = sum(position[1]) / len(position[0])
    data_l.append([round(length, 2), (round(x_cen,2), round(y_cen,2))])
  
title = ["length", "centroid"]
circle = list(np.arange(1, 7, 1))
print(pandas.DataFrame(data_l, circle, title))    


# i) The original image, the image after opening, the image after connected component labeling (plot with colorbar), and a table with the desired values for each component of part(ii) is shown above. 
# 
# ii) The structure used for the opening operation is rectangle and the size of it is width 1 with height 8.
# 
# iii) codes shown above
