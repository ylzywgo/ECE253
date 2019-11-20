import numpy as np
import imageio
import matplotlib.pyplot as plt

# ### Problem 4

# In[ ]:


def imageTrans(img):
    res1 = np.zeros(img.shape[0:2])
    res2 = np.zeros(img.shape)
    res3 = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j][1] > 120 and ((img[i][j][0] < 100 and img[i][j][2] < 180) 
                                        or (img[i][j][0] < 180 and img[i][j][2] < 100))):
                res1[i][j] = 0
                res2[i][j] = [0, 0, 0]
                res3[i][j] = [0, 0, 158]
            else:
                res1[i][j] = 255
                res2[i][j] = img[i][j]
                res3[i][j] = img[i][j]
    return res1.astype(int), res2.astype(int), res3.astype(int)


# In[ ]:


dog = imageio.imread('dog.jpg',pilmode="RGB")
travolta = imageio.imread('travolta.jpg',pilmode="RGB")
         
dog1, dog2, dog3 = imageTrans(dog)
travolta1, travolta2, travolta3 = imageTrans(travolta)

f2 = plt.figure(figsize=(14,8))
f2_ax1 = f2.add_subplot(231)
f2_ax2 = f2.add_subplot(232)
f2_ax3 = f2.add_subplot(233)
f2_ax4 = f2.add_subplot(234)
f2_ax5 = f2.add_subplot(235)
f2_ax6 = f2.add_subplot(236)

f2_ax1.imshow(dog1, cmap = plt.get_cmap('gray'))
f2_ax1.title.set_text("transform of dog.jpg for situation (i)")
f2_ax2.imshow(dog2)
f2_ax2.title.set_text("transform of dog.jpg for situation (ii)")
f2_ax3.imshow(dog3)
f2_ax3.title.set_text("transform of dog.jpg for situation (iii)")
f2_ax4.imshow(travolta1, cmap = plt.get_cmap('gray'))
f2_ax4.title.set_text("transform of travolta.jpg for situation (i)")
f2_ax5.imshow(travolta2)
f2_ax5.title.set_text("transform of travolta.jpg for situation (ii)")
f2_ax6.imshow(travolta3)
f2_ax6.title.set_text("transform of travolta.jpg for situation (iii)")
plt.show()
plt.savefig("problem4.png")
