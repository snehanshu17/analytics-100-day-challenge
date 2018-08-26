
# coding: utf-8

# imported the library

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# we have to import the basic and required libraries for our project and import the datasets as well as rename them.

# In[2]:


Train=pd.read_csv("train_c.csv")
Building_Ownership_Use=pd.read_csv("Building_Ownership_Use.csv")
Building_structure=pd.read_csv("Building_structure.csv")
res_train=pd.merge(Train,Building_Ownership_Use,on="building_id")
x=pd.merge(res_train,Building_structure,on="building_id")


# explore  the data

# In[15]:


#checking the datatypes
print('Training data shape: ', x.shape)
print('Training data types: ', x.dtypes)
x.head()


# Working with numerical values and check the missing value

# In[3]:


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = x.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[4]:


missing_values_table(x)


# In[5]:


#since the columns have missing values so for now we will drop both columns
x.drop(['count_families','has_repair_started'],axis=1,inplace=True)


# In[6]:


#check whether more missing values are there or not
print(missing_values_table(x))


# In[7]:


#Dropping unnesscary columns(check the correlation)
x.drop(['district_id_x', 'district_id_y', 'vdcmun_id_x', 'vdcmun_id_y', 'ward_id_y','building_id','ward_id_x','vdcmun_id'], axis=1, inplace=True)


# In[66]:


#x.drop(['ward_id_x','vdcmun_id'], axis=1, inplace=True)


# In[8]:


#change before and after earthquake
x['count_floors_change']=x['count_floors_post_eq']/x['count_floors_pre_eq']
x['height_ft_change'] = (x['height_ft_post_eq']/x['height_ft_pre_eq'])
x.drop(['count_floors_post_eq', 'height_ft_post_eq','count_floors_pre_eq','height_ft_pre_eq'], axis=1, inplace=True)


# In[33]:


#data visualisation
plt.figure(figsize=(20,30))
sns.countplot(x='district_id', hue='damage_grade',data=x)


# we can see the district 22 23 ,28,30 are most affected by the earthqauake

# In[39]:


plt.figure(figsize=(25,30))
sns.countplot(x='area_assesed', hue='damage_grade',data=x)


# In[40]:


plt.figure(figsize=(25,30))
sns.countplot(x='land_surface_condition', hue='damage_grade',data=x)


# Flat are more affected by the earthquake

# In[41]:


plt.figure(figsize=(25,30))
sns.countplot(x='foundation_type', hue='damage_grade',data=x)


# Building which are build by Mud mortar stone and brick are more affected by earthquake

# In[42]:


plt.figure(figsize=(25,30))
sns.countplot(x='roof_type', hue='damage_grade',data=x)


# In[43]:


plt.figure(figsize=(25,30))
sns.countplot(x='ground_floor_type', hue='damage_grade',data=x)


# In[44]:


plt.figure(figsize=(25,30))
sns.countplot(x='position', hue='damage_grade',data=x)


# In[45]:


plt.figure(figsize=(25,30))
sns.countplot(x='plan_configuration', hue='damage_grade',data=x)


# In[46]:


plt.figure(figsize=(25,30))
sns.countplot(x='other_floor_type', hue='damage_grade',data=x)


# In[47]:


plt.figure(figsize=(25,30))
sns.countplot(x='legal_ownership_status', hue='damage_grade',data=x)


# In[48]:


plt.figure(figsize=(25,30))
sns.countplot(x='condition_post_eq', hue='damage_grade',data=x)


# Working with categorical variable

# In[9]:


print(x.select_dtypes('object').nunique())


# In[10]:


y=x['damage_grade']
x.drop(['damage_grade'],axis=1,inplace=True)


# In[11]:


# one-hot encoding of categorical variables
x = pd.get_dummies(x,drop_first=True)
print('Training Features shape: ', x.shape)


# Work on finding anamalies in the dataset, check if any column has same number of unique vales as length of x or any outliers in the data

# In[12]:


x.describe()


# In[13]:


# Plot the distribution of ages in years
plt.hist(x['area_assesed_Building removed'], edgecolor = 'k', bins = 10)
plt.title('Area assessed'); plt.xlabel('Area'); plt.ylabel('Count');


# In[14]:


#building the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score


# In[15]:


# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[16]:


#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)


# In[17]:


ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")


# In[90]:


#Correlation 
#correlation map
f,ax = plt.subplots(figsize=(30, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# Univariate feature selection and random forest classification

# In[134]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# find best scored 5 features
select_feature = SelectKBest(chi2, k=75).fit(x_train, y_train)


# In[135]:


print('Score list:', select_feature.scores_)
print('Feature list:', x_train.columns)


# In[136]:


x_train_2 = select_feature.transform(x_train)
x_test_2 = select_feature.transform(x_test)


# In[137]:


clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)


# Recursive feature elimination (RFE) with random forest

# In[139]:


from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestClassifier()      
rfe = RFE(estimator=clf_rf_3, n_features_to_select=15, step=1)
rfe = rfe.fit(x_train, y_train)


# In[140]:


print('Chosen best 15 feature by rfe:',x_train.columns[rfe.support_])


#  Recursive feature elimination with cross validation and random forest classification

# In[141]:


from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])


# In[142]:


# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[143]:


clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()


# In[44]:



z=x[['plinth_area_sq_ft','count_floors_change','age_building','height_ft_change','district_id',
     'condition_post_eq_Damaged-Not used','condition_post_eq_Damaged-Repaired and used','condition_post_eq_Not damaged','condition_post_eq_Damaged-Used in risk',
       'condition_post_eq_Damaged-Rubble unclear','condition_post_eq_Damaged-Rubble clear','area_assesed_Building removed']]
z.head()


# In[41]:


x.columns


# In[45]:


#building the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score


# In[46]:


# split data train 70 % and test 30 %
z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.3, random_state=100)


# In[47]:


#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(z_train,y_train)


# In[48]:


ac = accuracy_score(y_test,clf_rf.predict(z_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(z_test))
sns.heatmap(cm,annot=True,fmt="d")

