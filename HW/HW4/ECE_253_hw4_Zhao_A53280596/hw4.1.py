

# In[16]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage


# In[17]:


def HughTransform(img):
#     H = dict()
    dAll = list()
    dmax = int(np.ceil(np.sqrt(np.square(img.shape[0]) + np.square(img.shape[1]))))
    H = np.zeros((2*dmax, 181))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y] == 255:
                dOne = list()
                for theta in range(-90, 91):
                    thetaReal = theta / 180 * np.pi
                    d = int(np.floor(x * np.cos(thetaReal) + y * np.sin(thetaReal)))
                    dOne.append(d)
                    H[d + dmax][theta + 90] += 1
                dAll.append(dOne)
    return H, dAll,dmax


# In[18]:


test_img = np.zeros((11, 11))
test_img[0][0] = 255
test_img[0][10] = 255
test_img[5][5] = 255
test_img[10][0] = 255
test_img[10][10] = 255
plt.imshow(test_img, cmap = 'gray')
plt.title("original image")
plt.savefig("original image for (ii)")
plt.show()


H, dAll,dmax = HughTransform(test_img)


# In[19]:


plt.imshow(H, cmap = 'gray', aspect = 'auto', extent = [-90, 90, -dmax, dmax])
plt.colorbar()
plt.title("HT")
plt.xlabel("theta")
plt.ylabel("rho")
plt.savefig("HT for (ii)")
plt.show()


# In[20]:


maxDandTheta = list()
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if H[i][j] > 2:
            maxDandTheta.append((i-dmax, j-90))
# print(maxDandTheta)


# In[21]:


def afterHughTransform(img, maxDandTheta):
    res_img = img.copy()
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for td in maxDandTheta:
                thetaReal = td[1] / 180 * np.pi
                d = td[0]

                d_new = int(np.round(x * np.cos(thetaReal) + y * np.sin(thetaReal)))
                if d_new == d:
                    res_img[x][y] = 255
    return res_img


# In[22]:


res_img = afterHughTransform(test_img, maxDandTheta)
plt.imshow(res_img, cmap = 'gray')
plt.title("original image with lines")
plt.savefig("original image with lines for (ii)")
plt.show()


# In[23]:


img = cv2.imread("data/lane.png", 0)
plt.imshow(img, cmap = 'gray')
plt.title("original image")
plt.savefig("original image for (iii)")
plt.show()

edges = cv2.Canny(img, 200, 200)
plt.imshow(edges, cmap='gray')
plt.title("binary edge image")
plt.savefig("binary edge image for (iii)")
plt.show()


# In[24]:


H, dAll,dmax = HughTransform(edges)
maxH = np.max(H)


# In[25]:


plt.imshow(H, cmap = 'gray', aspect = 'auto', extent = [-90, 90, -dmax, dmax])
plt.xlabel("theta")
plt.ylabel("rho")
plt.title("HT")
plt.colorbar()
plt.savefig("HT for (iii)")
plt.show()


# In[26]:


maxDandTheta = list()
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if H[i][j] > maxH * 0.75:
            maxDandTheta.append((i-dmax, j-90))
            
maxDandTheta_filtered = list()
for td in maxDandTheta:
    if td[1] in range(-38, - 34) or td[1] in range(34, 39):
        maxDandTheta_filtered.append(td)
        
# print(maxDandTheta_filtered)
# print(maxH * 0.75)


# In[27]:


def afterHughTransform_new(img, maxDandTheta):
    res_img = img.copy()
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for td in maxDandTheta:
                thetaReal = td[1] / 180 * np.pi
                d = td[0]

                d_new = int(np.round(x * np.cos(thetaReal) + y * np.sin(thetaReal)))
                if d_new == d:
                    res_img[x][y] = 0
    return res_img


# In[28]:


res_img = afterHughTransform_new(img, maxDandTheta)
plt.imshow(res_img, cmap = 'gray')
plt.title("original image with lines")
plt.savefig("original image with lines for (iii)")
plt.show()


# In[29]:


res_img = afterHughTransform_new(img, maxDandTheta_filtered)
plt.imshow(res_img, cmap = 'gray')
plt.title("original image with lines for (iv)")
plt.show()


# In[ ]:




