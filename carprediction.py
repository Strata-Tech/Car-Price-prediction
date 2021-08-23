#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('data/car data.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[7]:


#check for null values

df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df.columns


# In[10]:


final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[11]:


final_dataset.head()


# In[12]:


final_dataset['current_year']=2020


# In[13]:


final_dataset


# In[14]:


final_dataset['no_of_years']=final_dataset['current_year']-final_dataset['Year']


# In[15]:


final_dataset.head()


# In[16]:


final_dataset.drop(columns={'Year','current_year'},axis=1,inplace=True)


# In[17]:


final_dataset.head()


# In[18]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[19]:


final_dataset.head()


# In[20]:


pip install seaborn


# In[21]:


pip install matplotlib


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


sns.pairplot(final_dataset)


# In[26]:


corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))


# In[30]:


#plot heatmap
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[31]:


final_dataset.head()


# In[32]:


#independent features and dependent feature

X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[33]:


X.head()


# In[34]:


y.head()


# In[36]:


pip install sklearn


# In[37]:


#to find out which are the important features

from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)


# In[38]:


print(model.feature_importances_)


# In[40]:


#visualize feature importance

feature_importances=pd.Series(model.feature_importances_,index=X.columns)
feature_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[ ]:





# In[41]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[42]:


X_train.shape

pip install numpy 
# In[46]:


pip install numpy


# In[47]:


import numpy as np


# In[44]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()


# In[48]:


#Hyperparameter tuning/optimization
#Randomized SearchCV

n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
#number of features to consider at every split

max_features=['auto','sqrt']
#max number of levels in tree
max_depth= [int(x) for x in np.linspace(start=5,stop=30,num=6)]

# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[50]:


from sklearn.model_selection import RandomizedSearchCV


# In[51]:


# Create the random grid in a dictionary of key , value pairs
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[52]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[53]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[54]:


rf_random.fit(X_train,y_train)


# In[55]:


rf_random.best_params_


# In[56]:


rf_random.best_score_


# In[57]:


predictions=rf_random.predict(X_test)


# In[58]:


sns.distplot(y_test-predictions)


# In[59]:


plt.scatter(y_test,predictions)


# In[60]:


from sklearn import metrics


# In[61]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[62]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:





# In[ ]:





# In[ ]:




