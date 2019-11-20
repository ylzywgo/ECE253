import numpy as np

A = np.array([[3, 9, 5, 1], 
              [4, 25, 4, 3], 
              [63, 13, 23, 9], 
              [6, 32, 77, 0], 
              [12, 8, 6, 1]])

B = np.array([[0, 1, 0, 1],
              [0, 1, 1, 0], 
              [0, 0, 0, 1], 
              [1, 1, 0, 1], 
              [0, 1, 0, 0]])


# In[ ]:


C = A * B
print(C)


# In[ ]:


np.dot(C[1,:], C[3,:])


# In[ ]:


minn = np.min(C)
maxn = np.max(C)


# In[ ]:


print(minn, maxn)


# In[ ]:


minindex = np.where(C == minn)
maxindex = np.where(C == maxn)
print(minindex)
print(maxindex)


