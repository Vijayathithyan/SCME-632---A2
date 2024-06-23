#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Import libraries
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
import os
import pandas as pd


# In[10]:


# Set working directory
os.chdir('D:\\MDA\\Course\\Boot Camp\\SCMA 632\\Assignments\\A2')

# Load dataset
data = pd.read_csv("NSSO68.csv")

# Subset data to state assigned
subset_data = data[data['state_1'] == 'Bhr'][['foodtotal_q', 'MPCE_MRP', 
                                              'MPCE_URP', 'Age', 'Meals_At_Home', 
                                              'Possess_ration_card', 'Education', 
                                              'No_of_Meals_per_day']]

# Check for missing values
print(subset_data.isnull().sum())


# In[11]:


# Define function to impute missing values with mean
def impute_with_mean(data, columns):
  return data.apply(lambda x: x.fillna(x.mean()), axis=0)

# Impute missing values in Education column
subset_data = impute_with_mean(subset_data, ['Education'])

# Check for missing values after imputation
print(subset_data.isnull().sum())


# In[13]:


# Fit the regression model
model = ols('foodtotal_q ~ MPCE_MRP + MPCE_URP + Age + Meals_At_Home + Possess_ration_card + Education', data=subset_data).fit()

# Print the regression results
print(model.summary())


# In[5]:


pip install statsmodels


# In[14]:


# Extract coefficients
coefficients = model.params


# In[15]:


# Construct the equation
equation = "y = " + str(round(coefficients[0], 2))
for i in range(1, len(coefficients)):
  equation += " + " + str(round(coefficients[i], 6)) + "*x" + str(i)

# Print the equation
print(equation)


# In[16]:


# Print the first element of each variable (for checking purposes)
print(subset_data['MPCE_MRP'].head(1))
print(subset_data['MPCE_URP'].head(1))
print(subset_data['Age'].head(1))
print(subset_data['Meals_At_Home'].head(1))
print(subset_data['Possess_ration_card'].head(1))
print(subset_data['Education'].head(1))
print(subset_data['foodtotal_q'].head(1))

