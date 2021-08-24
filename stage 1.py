#!/usr/bin/env python
# coding: utf-8

# In[1]:


A = [1,2,3,4,5,6], B = [13, 21, 34]


# In[5]:


A = [1,2,3,4,5,6]
B = [13, 21, 34]


# In[6]:


A.extend(B)
print(A)


# In[7]:


import numpy as np
np.identity(3)


# In[9]:


import pandas as pd
data=pd.read_csv("https://raw.githubusercontent.com/WalePhenomenon/climate_change/master/fuel_ferc1.csv")


# In[30]:


data.head(25)#.describe()


# In[21]:


data.groupby(["fuel_type_code_pudl"]).sum()


# In[24]:


data.isna().sum()


# In[25]:


data["fuel_unit"].isna().sum()/len(data.index)


# In[27]:


len(data.index)


# In[54]:


data.groupby(["report_year"]).sum()


# In[66]:


dat2["fuel_cost_per_unit_delivered"].max()


# In[75]:


print(dat2[dat2["fuel_cost_per_unit_delivered"]==dat2["fuel_cost_per_unit_delivered"].max()].index.values)


# In[99]:


print(data[['report_year','fuel_cost_per_unit_delivered',"fuel_type_code_pudl"]].groupby(["report_year","fuel_type_code_pudl"]).mean().head(50
                                                                                                                                     ))


# In[81]:


dat3


# In[ ]:




