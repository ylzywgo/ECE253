import numpy as np
import imageio
import matplotlib.pyplot as plt

A = imageio.imread('111.jpg')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144]).astype(int)

B = rgb2gray(A)
minn = np.min(B)
maxn = np.max(B)
print("min of B is", minn, ". max of B is", maxn)

C = B + 15
C[C > 255] = 255

D = np.flip(np.flip(C, 0),1)

median = np.median(B)
E = np.zeros(B.shape)
small = B <= median
large = B > median
E[small] = 1
E[large] = 0

f1 = plt.figure(figsize=(12,6))
f1_ax1 = f1.add_subplot(231)
f1_ax2 = f1.add_subplot(232)
f1_ax3 = f1.add_subplot(233)
f1_ax4 = f1.add_subplot(234)
f1_ax5 = f1.add_subplot(235)

f1_ax1.imshow(A)
f1_ax1.title.set_text("A")
f1_ax2.imshow(B, cmap = plt.get_cmap('gray'))
f1_ax2.title.set_text("B")
f1_ax3.imshow(C, cmap = plt.get_cmap('gray'))
f1_ax3.title.set_text("C")
f1_ax4.imshow(D, cmap = plt.get_cmap('gray'))
f1_ax4.title.set_text("D")
f1_ax5.imshow(E, cmap = plt.get_cmap('gray'))
f1_ax5.title.set_text("E")
plt.show()
plt.savefig("problem2.png")
