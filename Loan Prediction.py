#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("radhe radhe")


# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


dataset = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv") 


# In[5]:


dateset.head()


# In[6]:


dataset.head()


# In[12]:


dataset.shape


# In[9]:


dataset.info()


# In[13]:


dataset.describe()


# In[16]:


pd.crosstab(dataset['Credit_History'], dataset['Loan_Amount_Term'], margins=True)


# In[18]:


pd.crosstab(dataset['Credit_History'], dataset['LoanAmount'], margins=True)


# In[20]:


dataset.LoanAmount


# In[21]:


dataset.Loan_Status


# In[26]:


dataset.boxplot(column='ApplicantIncome')


# In[27]:


dataset['ApplicantIncome'].hist(bins=20)


# In[28]:


dataset['CoapplicantIncome'].hist(bins=20)


# In[29]:


dataset.boxplot(column= 'ApplicantIncome' ,by= 'Education')


# In[30]:


dataset.boxplot(column= 'ApplicantIncome' ,by= 'Gender')


# In[31]:


dataset.boxplot(column='LoanAmount')


# In[32]:


dataset['LoanAmount'].hist(bins=20)


# In[33]:


dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[34]:


dataset.isnull().sum()


# In[15]:


dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)


# In[16]:


dataset['Married'].fillna(dataset['Married'].mode(),inplace=True)


# In[17]:


dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)


# In[18]:


dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)


# In[20]:


dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())


# In[21]:


dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)


# In[22]:


dataset.isnull().sum()


# In[64]:


dataset.isnull().sum()


# In[10]:


dataset.head()


# In[9]:


dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)


# In[11]:


dataset = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv") 


# In[12]:


dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)


# In[13]:


dataset.head()


# In[23]:


dataset.isnull().sum()


# In[24]:


dataset.boxplot(column= 'ApplicantIncome' ,by= 'Gender')


# In[25]:


dataset.boxplot(column= 'ApplicantIncome' ,by= 'Married')


# In[30]:


dataset.boxplot(column= 'ApplicantIncome' ,by= 'Married')


# In[31]:


dataset.boxplot(column= 'ApplicantIncome' ,by= 'Education')


# In[34]:


dataset.boxplot(column= 'LoanAmount' ,by= 'Married')


# In[35]:


dataset['TotalIncome']=dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome_log']= np.log(dataset['TotalIncome'])


# In[37]:


dataset['TotalIncome_log'].hist(bins=20)


# In[38]:


dataset.head()


# In[61]:


X= dataset.iloc[:,np.r_[1:5,9:11,13:14]].values
Y= dataset.iloc[:,10].values 


# In[43]:


X


# In[62]:


Y


# In[70]:


from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train ,Y_test = train_test_split(X,Y, test_size= 0.2,random_state=0)


# In[71]:


print(X_train)


# In[72]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()


# In[73]:


for i in range(0, 5):
    X_train[:,i]= labelencoder_X.fit_transform(X_train[:,i])


# In[74]:





# In[75]:


X_train


# In[76]:


labelencoder_Y=LabelEncoder()
Y_train= labelencoder_Y.fit_transform(Y_train)


# In[77]:


Y_train


# In[78]:


for i in range(0, 5):
    X_test[:,i]= labelencoder_X.fit_transform(X_test[:,i])


# In[79]:


X_test[:,6]= labelencoder_X.fit_transform(X_test[:,6])


# In[80]:


X_test


# In[81]:


labelencoder_Y=LabelEncoder()
Y_test= labelencoder_Y.fit_transform(Y_test)


# In[82]:


Y_test


# In[85]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()


# In[88]:


X_train = ss.fit_transform(X_train)


# In[89]:


X_test=ss.fit_transform(X_test)


# In[90]:


from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
DTC.fit(X_train,Y_train)


# In[91]:


Y_pred= DTC.predict(X_test)
Y_pred


# In[95]:


from sklearn import metrics
print('The accuracy of Decision tree is:' , metrics.accuracy_score(Y_pred,Y_test))


# In[97]:


from sklearn.naive_bayes import GaussianNB
NBClassifer = GaussianNB()
NBClassifier.fit(X_train,Y_train)


# In[98]:


from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(X_train,Y_train)


# In[99]:


Y_pred= NBClassifier.predict(X_test)


# In[100]:


Y_pred


# In[104]:


print('The accuracy of naiveBayes is', metrics.accuracy_score(Y_pred,Y_test))


# In[105]:


pred = NBClassifier.predict(test)


# In[106]:


pred


# In[ ]:




