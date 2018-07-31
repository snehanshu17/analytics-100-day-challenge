
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('USA_Housing.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[10]:


sns.pairplot(df)


# In[11]:


sns.distplot(df['Price'])


# In[14]:


sns.heatmap(df.corr(),annot=True)


# In[15]:


df.columns


# In[25]:


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[26]:


Y=df['Price']


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)


# In[29]:


from sklearn.linear_model import LinearRegression


# In[30]:


lm=LinearRegression()


# In[31]:


lm.fit(X_train,Y_train)


# In[32]:


print(lm.intercept_)


# In[34]:


lm.coef_


# In[38]:


cdf=pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[39]:


cdf


# In[40]:


Predictions=lm.predict(X_test)


# In[41]:


Predictions


# In[43]:


plt.scatter(Y_test,Predictions)


# In[45]:


sns.distplot(Y_test-Predictions)


# In[46]:


from sklearn import metrics


# In[47]:


metrics.mean_absolute_error(Y_test,Predictions)


# In[49]:


np.sqrt(metrics.mean_absolute_error(Y_test,Predictions))

