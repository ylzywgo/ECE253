# ### Problem 5

# In[ ]:


import glob
import cv2
import matplotlib.pyplot as plt

images = [plt.imread(file) for file in glob.glob("interpolation/*.jpg")]
scale_percent = [0.3, 0.5, 0.7]

i = 1
for image in images:
    ref = plt.figure(figsize=(14,16))
    reax0 = ref.add_subplot(431)
    reax0.imshow(image)   
    reax0.title.set_text("original image")
#     plt.show()
    na = 4;
    for scale in scale_percent:
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        nearest = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)
        linear = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
        bicubic = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
        ax = ref.add_subplot(4,3,na)
        ax2 = ref.add_subplot(4,3,na+1)
        ax3 = ref.add_subplot(4,3,na+2)
        na += 3
        ax.title.set_text("nearest neighbor with downsampling ratio %s"%scale)
        ax.imshow(nearest)
        ax2.title.set_text("bilinear with downsampling ratio %s"%scale)
        ax2.imshow(linear)
        ax3.title.set_text("bicubic with downsampling ratio %s"%scale)
        ax3.imshow(bicubic)
    plt.savefig("question3_%s"%i)
    i += 1


# In[ ]:

images = [plt.imread(file) for file in glob.glob("interpolation/*.jpg")]
scale_percent = [1.5, 1.7, 2.0]

i = 1
for image in images:
    ref = plt.figure(figsize=(12,16))
    reax0 = ref.add_subplot(431)
    reax0.imshow(image)   
    reax0.title.set_text("original image")
#     plt.show()
    na = 4;
    for scale in scale_percent:
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        nearest = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)
        linear = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
        bicubic = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
        ax = ref.add_subplot(4,3,na)
        ax2 = ref.add_subplot(4,3,na+1)
        ax3 = ref.add_subplot(4,3,na+2)
        na += 3
        ax.title.set_text("nearest neighbor with upsampling ratio %s"%scale)
        ax.imshow(nearest)
        ax2.title.set_text("bilinear with upsampling ratio %s"%scale)
        ax2.imshow(linear)
        ax3.title.set_text("bicubic with upsampling ratio %s"%scale)
        ax3.imshow(bicubic)
    plt.savefig("question4_%s"%i)
    i += 1


# In[ ]:


images = [plt.imread(file) for file in glob.glob("interpolation/*.jpg")]

i = 1

for image in images:
    ref = plt.figure(figsize=(12,8))
    reax0 = ref.add_subplot(231)
    reax0.imshow(image)   
    reax0.title.set_text("original image")
#     plt.show()
    
    scale = 0.1
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    ori_dim = (image.shape[1], image.shape[0])
    
    nearest = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)
    linear = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
    bicubic = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    
    renearest = cv2.resize(nearest, ori_dim, interpolation = cv2.INTER_NEAREST)
    relinear = cv2.resize(linear, ori_dim, interpolation = cv2.INTER_LINEAR)
    rebicubic = cv2.resize(bicubic, ori_dim, interpolation = cv2.INTER_CUBIC)
    reax = ref.add_subplot(234)
    reax2 = ref.add_subplot(235)
    reax3 = ref.add_subplot(236)
    reax.title.set_text("nearest neighbor")
    reax.imshow(renearest)
    reax2.title.set_text("bilinear")
    reax2.imshow(relinear)
    reax3.title.set_text("bicubic")
    reax3.imshow(rebicubic)
#     plt.show()
    plt.savefig("question5_%s.jpg"%i)
    i += 1


# In[ ]:




