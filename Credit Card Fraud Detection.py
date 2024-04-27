#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


# In[4]:


data = pd.read_csv(r"D:\python datasets\creditcard.csv")


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.describe()


# In[8]:


data.info()


# In[10]:


fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

Fractional_value = len(fraud)/len(valid)
print(Fractional_value)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))


# In[12]:


fraud.Amount.describe()


# In[13]:


valid.Amount.describe()


# In[14]:


correlation = data.corr()
plotting = plt.figure(figsize =(15,9))
sns.heatmap(correlation,vmax = .10, square = True)
plt.show()


# In[15]:


x = data.drop(['Class'],axis = 1)
y = data['Class']
print(x.shape)
print(y.shape)


xData = x.values
yData = y.values


# In[32]:


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size =0.2, random_state =42)


# In[33]:


from sklearn.ensemble import RandomForestClassifier


# In[34]:


rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

yPred = rfc.predict(xTest)


# In[30]:


from skleran.metrics import classification_report, accuracy_score
from skleran.metrics import precision_score, recall_score
from skleran.metrics import f1_score, matthews_corrcoef
from skleran.metrics import confusion_matrix


# In[35]:


n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()
print("The model used is Random Forest Classifier")


# In[36]:


acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc))


# In[37]:


prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec))


# In[40]:


rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec))


# f1 = f1_score(yTest, yPred)
# print("The F1-score is{}".format(f1))

# In[41]:


f1 = f1_score(yTest, yPred)
print("The F1-score is{}".format(f1))


# In[42]:


MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is{}".format(MCC))


# In[44]:


LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize =(8,8))
sns.heatmap(conf_matrix, xticklabels = LABELS,
           yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[ ]:




