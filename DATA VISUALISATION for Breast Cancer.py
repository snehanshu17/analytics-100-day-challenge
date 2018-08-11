
# coding: utf-8

# #Breast Cancer detection 

# 
# INTRODUCTION
# 
# In this data analysis report, I usually focus on feature visualization and selection as a different 
# from other kernels. Feature selection with correlation, univariate feature selection, recursive feature elimination, recursive feature elimination with cross validation and tree based feature selection methods are used with random forest classification. Apart from these, 
# principle component analysis are used to observe number of components.

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


Cancer=pd.read_csv('Cancer.csv')


# In[8]:


Cancer.head()


# In[9]:


Cancer.info()


# There are 4 things that take my attention 1) There is an id that cannot be used for classificaiton 2) Diagnosis is our class label 3) Unnamed: 32 feature includes NaN so we do not need it. 4) I do not have any idea about other feature names actually I do not need because machine learning is awesome :)
# 
# Therefore, drop these unnecessary features. However do not forget this is not a feature selection. This is like a browse a pub, we do not choose our drink yet !!!

# In[10]:


Cancer.describe()


# In[11]:


# feature names as a list
col = Cancer.columns       # .columns gives columns names in data 
print(col)


# Drop the unnecessary columns

# In[12]:


list = ['Unnamed: 32','id','diagnosis']
X=Cancer.drop(list,axis=1)
Y=Cancer.diagnosis


# In[13]:


X.head()


# In[14]:


Y.head()


# Categorical  Plot

# In[15]:


ax=sns.countplot(Y,Label="Count")
B, M = Y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)


# Swarmplot

# In[16]:


data_dia=Y
data=X
data_n_2=(data-data.mean())/data.std()   #standarization


# In[17]:


data_n_2.describe()


# In[18]:


data = pd.concat([Y,data_n_2.iloc[:,0:10]],axis=1)


# This function is useful to massage a DataFrame into a format where one or more columns are identifier variables (id_vars), while all other columns, considered measured variables (value_vars), are “unpivoted” to the row axis, leaving just two non-identifier columns, ‘variable’ and ‘value’

# In[19]:


data=pd.melt(data,id_vars="diagnosis",
                  var_name="features",
                   value_name='value')

plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis',data=data,split=True,inner="quart")
plt.xticks(rotation=90)


# In[20]:


# Second ten features
data = pd.concat([Y,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)


# In[21]:


# Second ten features
data = pd.concat([Y,data_n_2.iloc[:,20:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)


# In[22]:


# As an alternative of violin plot, box plot can be used
# box plots are also useful in terms of seeing outliers
# I do not visualize all features with box plot
# In order to show you lets have an example of box plot
# If you want, you can visualize other features as well.

plt.figure(figsize=(10,10))
sns.boxplot(x='features',y='value',hue='diagnosis',data=data) 
plt.xticks(rotation=90)


# In order to compare two features deeper, lets use joint plot. Look at this in joint plot below, it is really correlated. Pearsonr value is correlation value and 1 is the highest. Therefore, 0.86 is looks enough to say that they are correlated. Do not forget, we are not choosing features yet, we are just looking to have an idea about them.

# In[23]:


sns.jointplot(X.loc[:,'concavity_worst'], X.loc[:,'concave points_worst'], kind="regg", color="#ce1414")


# In[24]:


u=X.corr()


# In[68]:


f,ax = plt.subplots(figsize=(18, 18))         # Sample figsize in inches
sns.heatmap(u,annot=True, linewidths=.5,fmt='.1f', ax=ax)


# What about three or more feauture comparision ? For this purpose we can use pair grid plot. Also it seems very cool :) And we discover one more thing radius_worst, perimeter_worst and area_worst are correlated as it can be seen pair grid plot. We definetely use these discoveries for feature selection.

# In[31]:


sns.set(style="white")
df = X.loc[:,['radius_worst','perimeter_worst','area_worst']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)


# In[63]:


data_dia =Y
data = X
data_n_2 = (data - data.mean()) / (data.std()) 
data = pd.concat([Y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# In[64]:


data_dia =Y
data = X
data_n_2 = (data - data.mean()) / (data.std()) 
data = pd.concat([Y,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# In[65]:


data_dia =Y
data = X
data_n_2 = (data - data.mean()) / (data.std()) 
data = pd.concat([Y,data_n_2.iloc[:,20:30]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
#tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# 1) Feature selection with correlation and random forest classification
# As it can be seen in map heat figure radius_mean, perimeter_mean and area_mean are correlated with each other so we will use only area_mean. If you ask how i choose area_mean as a feature to use, well actually there is no correct answer, I just look at swarm plots and area_mean looks like clear for me but we cannot make exact separation among other correlated features without trying. So lets find other correlated features and look accuracy with random forest classifier.
# 
# Compactness_mean, concavity_mean and concave points_mean are correlated with each other.Therefore I only choose concavity_mean. Apart from these, radius_se, perimeter_se and area_se are correlated and I only use area_se. radius_worst, perimeter_worst and area_worst are correlated so I use area_worst. Compactness_worst, concavity_worst and concave points_worst so I use concavity_worst. Compactness_se, concavity_se and concave points_se so I use concavity_se. texture_mean and texture_worst are correlated and I use texture_mean. area_worst and area_mean are correlated, I use area_mean.

# In[71]:


drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se',
              'perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst',
              'compactness_se','concave points_se','texture_worst','area_worst']
x_1 = X.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 
x_1.head()


# In[75]:


f,ax = plt.subplots(figsize=(14, 14))         # Sample figsize in inches
sns.heatmap(x_1.corr(),annot=True, linewidths=.5,fmt='.1f', ax=ax)


# In[76]:


#Random Forest

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score


# In[78]:


# split data train 70 % and test 30 %
X_train, X_test, Y_train, Y_test = train_test_split(x_1, Y, test_size=0.3, random_state=42)


# In[79]:


#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(X_train,Y_train)


# In[82]:


ac = accuracy_score(Y_test,clf_rf.predict(X_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(Y_test,clf_rf.predict(X_test))
sns.heatmap(cm,annot=True,fmt="d")


# 2) Univariate feature selection and random forest classification

# In this method we need to choose how many features we will use. For example, will k (number of features) be 5 or 10 or 15? The answer is only trying or intuitively. I do not try all combinations but I only choose k = 5 and find best 5 features.

# In[87]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# find best scored 5 features
select_feature = SelectKBest(chi2, k=5).fit(X_train, Y_train)


# In[86]:


print('Score list:', select_feature.scores_)
print('Feature list:', X_train.columns)


# 
# Best 5 feature to classify is that area_mean, area_se, texture_mean, concavity_worst and concavity_mean. So lets se what happens if we use only these best scored 5 feature.

# In[88]:


x_train_2 = select_feature.transform(X_train)
x_test_2 = select_feature.transform(X_test)


# In[90]:


#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train_2,Y_train)


# In[92]:


ac_2 = accuracy_score(Y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(Y_test,clf_rf_2.predict(x_test_2))
sns.heatmap(cm_2,annot=True,fmt="d")


# Accuracy is almost 93% and as it can be seen in confusion matrix, we make few wrong prediction. What we did up to now is that we choose features according to correlation matrix and according to selectkBest method. Although we use 5 features in selectkBest method accuracies look similar. Now lets see other feature selection methods to find better results.

# 3) Recursive feature elimination (RFE) with random forest¶
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html Basically, it uses one of the classification methods (random forest in our example), assign weights to each of features. Whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features

# Like previous method, we will use 5 features. However, which 5 features will we use ? We will choose them with RFE method.

# In[93]:


from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestClassifier()      
rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
rfe = rfe.fit(X_train, Y_train)


# In[95]:


print('Chosen best 5 feature by rfe:',X_train.columns[rfe.support_])


# 4) Recursive feature elimination with cross validation and random forest classification

# In[96]:


from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, Y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])


# Finally, we find best 11 features that are texture_mean, area_mean, concavity_mean, texture_se, area_se, concavity_se, symmetry_se, smoothness_worst, concavity_worst, symmetry_worst and fractal_dimension_worst for best classification. Lets look at best accuracy with plot.

# In[97]:


# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# 5) Tree based feature selection and random forest classification
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html In random forest classification method there is a featureimportances attributes that is the feature importances (the higher, the more important the feature). !!! To use feature_importance method, in training data there should not be correlated features. Random forest choose randomly at each iteration, therefore sequence of feature importance list can change.

# In[102]:


clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(X_train,Y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# In[103]:


# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices],rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[105]:




