import numpy as np
import imageio
import matplotlib.pyplot as plt

# ### Problem 3

# In[ ]:


def compute_norm_rgb_histogram(A):
    R = []
    G = []
    B = []
    RA = A[:,:,0]
    GA = A[:,:,1]
    BA = A[:,:,2]
    for i in range(32):
        Rnum = RA[RA >= i * 8]
        Rnum = Rnum[Rnum <= (i+1) * 8 -1]
        R.append(len(Rnum))
        
        Gnum = GA[GA >= i * 8]
        Gnum = Gnum[Gnum <= (i+1) * 8 -1]
        G.append(len(Gnum))
        
        Bnum = BA[BA >= i * 8]
        Bnum = Bnum[Bnum <= (i+1) * 8 -1]
        B.append(len(Bnum))
    RGB = R + G + B
    sumRGB = sum(RGB)
    RGB = [x / sumRGB for x in RGB] 
    
    return RGB


# In[ ]:


image = imageio.imread('geisel.jpg',pilmode="RGB")
RGB = compute_norm_rgb_histogram(image)
plt.bar(range(0, 96), RGB)
plt.title("Normalized RGB Histogram")
plt.show()
plt.savefig("problem3.png")
print("Sum of the histogram is",sum(RGB))


