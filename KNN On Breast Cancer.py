#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dfx=pd.read_csv("data.csv")
print(dfx)


# In[11]:


x=dfx.drop(columns="id")
x=x.drop(columns="texture_mean")
x=x.drop(columns="perimeter_mean")
x=x.drop(columns="area_mean")
print(x)


# In[12]:


x=x.values
print(x)


# In[13]:


print(x.shape)


# In[61]:


Y=x[:,0]
X=x[:,1:3]
print(X)
print(Y)


# In[73]:


plt.scatter(X[:,0],X[:,1],c=Y)
queryx=np.array([15,0.10])
plt.scatter(queryx[0],queryx[1],color="red")
plt.show()


# 

# In[74]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))


# In[75]:


def knn(X,Y,queryx,k=5):
    vals=[]
    m=X.shape[0]
    print(m)
    for i in range(m):
        d=dist(X[i],queryx)
        vals.append((d,Y[i]))
        
    vals=sorted(vals)
    vals=vals[:k]
    vals=np.array(vals)
    new=np.unique(vals[:,1],return_counts=True)
    index=new[1].argmax()
    pred=new[0][index]
    print(pred)


# In[76]:


knn(X,Y,queryx)


# In[ ]:





# In[ ]:




