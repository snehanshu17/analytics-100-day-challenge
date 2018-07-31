
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as py


# In[13]:


df=pd.read_csv('txt.txt')


# In[16]:


x=df["population of a city"]
y=df["profit of a food truck"]


# In[18]:


y.head()


# In[45]:


py.plot(x,y,"x")
py.title("scatter plot for simple linear regression")
py.xlabel("population of a city")
py.ylabel("the profit of a food truck in that city")
py.show()
type(x)


# In[46]:


iterations = 1500;
alpha = 0.01;
x0 = np.ones(np.size(x))
X=np.array([x0,x]).T


# In[47]:


def cost(X,y,theta=[[0],[0]]):
    m=y.size
    h=X.dot(theta)
    return (np.sum(np.square((h - y)))/(2 * m))


# In[50]:


theta=np.array([0,0]) #weights


# In[55]:


cost(X,y,theta)


# In[56]:


#gradient descent
def gradient(X,y,theta,alpha,iterations):
    cost_hist=np.zeros(iterations)
    for i in range(iterations):
        h=X.dot(theta)
        theta=theta-alpha*((X.T.dot(h-y))/y.size)
        cost_hist[i]=cost(X,y,theta)
    return theta,cost_hist


# In[58]:


final_theta,history=gradient(X,y,theta,alpha,iterations)
final_theta


# In[59]:


py.plot(history)
py.ylabel('Cost J')
py.xlabel('Iterations');


# In[62]:


from  sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1), y.ravel())
normal_x=np.arange(5,23)
pred_y=final_theta[0]+normal_x.dot(final_theta[1])
py.plot(normal_x,pred_y,label="model form scratch")
py.plot(normal_x, regr.intercept_+regr.coef_*normal_x, label='Linear regression (Scikit-learn GLM)')
py.show()

