#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing the required library
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 12))
import seaborn as sns
sns.set(color_codes=True)


# In[8]:


# loading the dataset
df1 = pd.read_csv(r'C:\Users\Tripple A\Downloads\energydata_complete.csv')
df1.head()


# In[9]:


df1.describe()


# In[10]:


df1.duplicated().any()
df1.isnull().sum()


# In[11]:



df2 = df1.drop(columns=["date", "lights"], axis=1)
df2


# In[12]:


x = df2["T2"]
y = df2["T6"]


# In[13]:


array = np.array([x])
x_feature =  array.reshape(-1, 1)


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_feature, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
predicted_value = linear_model.predict(X_test)


# In[15]:


# mean absolute error
from sklearn.metrics import mean_absolute_error
mae_reg = mean_absolute_error(y_test, predicted_value)
round(mae_reg, 2)


# In[16]:


# residual sum of squares (RSS)
import numpy as np
rss_reg = np.sum(np.square(y_test - predicted_value))
round(rss_reg, 2)


# In[22]:


# Root Mean Square Error (RMSE)
from sklearn.metrics import mean_squared_error
rmse_reg = np.sqrt(mean_squared_error(y_test, predicted_value))
round(rmse_reg, 3)


# In[ ]:





# In[ ]:




