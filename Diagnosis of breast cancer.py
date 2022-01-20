#!/usr/bin/env python
# coding: utf-8

# #  Diagnosis of breast cancer cells

# ## Library imports and Data retrival

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


# In[2]:


df = pd.read_csv("data.csv")


# In[3]:


df.head()


# ## Cleaning the data
# 

# In[4]:


# Dropping the id column


# In[5]:


df = df.drop(columns='id', axis = 1)


# In[6]:


# Dropping the unnamed column


# In[7]:


df = df.drop(columns='Unnamed: 32', axis = 1)


# In[8]:


df.head()


# In[9]:


# Mapping the diagnosis values M and B as 1 and 0 respectivly


# In[10]:


df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})


# ## Grouping input variables and exploring correlations

# In[11]:


# Grouping all the data by the type of mean, standard error and worst features


# In[12]:


mean_features = list(df.columns[1:11])
se_features = list(df.columns[11:21])
worst_features = list(df.columns[21:31])


# In[13]:


# Appending diagnosis to do them such that a correlation can be calculated.


# In[14]:


mean_features.append('diagnosis')
se_features.append('diagnosis')
worst_features.append('diagnosis')


# In[15]:


# Creating a correlation matrix for our grouped data


# In[16]:


corr = df[mean_features].corr()
corr


# In[17]:


corr = df[se_features].corr()
corr


# In[18]:


corr = df[worst_features].corr()
corr


# In[19]:


# We can use the correlation matrix to identify the different varible that we should use in our model


# In[20]:


prediction_vars = ['radius_mean', 'perimeter_mean','area_mean','compactness_mean','concavity_mean',
                  'concave points_mean','radius_se', 'radius_worst', 'perimeter_worst', 'area_worst', 'compactness_worst']


# # Training the Model

# ### Spliting the dataset into train and test

# In[21]:


train, test = train_test_split(df, test_size = 0.15, random_state=1)


# In[22]:


train_x = train[prediction_vars]
train_y = train['diagnosis']
test_x = test[prediction_vars]
test_y = test['diagnosis']


# ## K Nearest Neighbour 

# In[23]:


#Initializing a KNN Model
modelKNN = KNeighborsClassifier()


# In[24]:


#Fitting the data into the model
modelKNN.fit(train_x, train_y)


# In[25]:


#Predicting y value using the test x data
predictionsKNN = modelKNN.predict(test_x)


# In[26]:


#initalizing a confusion matrix
matrixKNN = confusion_matrix(test_y, predictionsKNN, labels=[0,1])


# In[27]:


#Calculating the precision, recall and accuracy

precision_KNN = precision_score(test_y, predictionsKNN)
recall_KNN = recall_score(test_y, predictionsKNN)
accuracy_KNN = accuracy_score(test_y, predictionsKNN)


# ## Decision Tree

# In[28]:


#Initializing a Decision Tree Model
DecisionTreeModel =  DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=33)


# In[29]:


#Fitting the data into the model
DecisionTreeModel.fit(train_x, train_y)


# In[30]:


#Predicting y value using the test x data
predictionsDecisionTree = DecisionTreeModel.predict(test_x)


# In[31]:


#initalizing a confusion matrix
matrixDT = confusion_matrix(test_y, predictionsDecisionTree, labels=[0,1])


# In[32]:


#Calculating the precision, recall and accuracy

precision_DecisionTree = precision_score(test_y, predictionsDecisionTree)
recall_DecisionTree = recall_score(test_y, predictionsDecisionTree)
accuracy_DecisionTree = accuracy_score(test_y, predictionsDecisionTree)


# In[33]:


models=['Decision Tree', 'K Nearest Neighbour']
prec = [precision_DecisionTree,precision_KNN]
acc = [accuracy_DecisionTree, accuracy_KNN]
rec = [recall_DecisionTree,recall_KNN]


# ## Comparing the model results

# In[34]:


data = zip(models,acc, prec, rec)


# In[35]:


result = pd.DataFrame(data,columns=['Model','Accuracy','Precision','Recall']).sort_values(["Accuracy"], ascending = False)


# In[36]:


result


# In[37]:


plt.figure(figsize=(15,7))
sns.barplot(x = "Model", y = "Accuracy", data = result)
plt.show()


# In[38]:


plt.figure(figsize=(15,7))
sns.barplot(x = "Model", y = "Recall", data = result)
plt.show()


# In[39]:


plt.figure(figsize=(15,7))
sns.barplot(x = "Model", y = "Precision", data = result)
plt.show()


# In[40]:


# Plot non-normalized confusion matrix for KNN model
cat = ['Benign - 0', 'Malignant - 1']
ax = plt.axes()
ax.set_title('Confusion Matrix for KNN Model')
sns.heatmap(matrixKNN,  annot=True, xticklabels= cat ,yticklabels= cat,  cmap='Blues', ax = ax)


# In[41]:


# Plot non-normalized confusion matrix for Decision Tree model
ax2 = plt.axes()
ax2.set_title('Confusion Matrix for Decision Tree Model')
sns.heatmap(matrixDT, annot=True, xticklabels= cat ,yticklabels= cat,  cmap='Reds', ax = ax2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




