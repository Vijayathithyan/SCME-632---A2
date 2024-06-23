#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[2]:


os.chdir('D:\MDA\Course\Boot Camp\SCMA 632\Assignments\A2')


# In[3]:


df_ipl = pd.read_csv('IPL_ball_by_ball_updated till 2024.csv')
salary = pd.read_excel('IPL SALARIES 2024.xlsx')


# In[4]:


df_ipl.columns


# In[5]:


grouped_data = df_ipl.groupby(['Season', 'Innings No', 'Striker','Bowler']).agg({'runs_scored': sum, 'wicket_confirmation':sum}).reset_index()


# In[6]:


grouped_data


# In[7]:


total_runs_each_year = grouped_data.groupby(['Season', 'Striker'])['runs_scored'].sum().reset_index()
total_wicket_each_year = grouped_data.groupby(['Season', 'Bowler'])['wicket_confirmation'].sum().reset_index()


# In[8]:


total_runs_each_year


# In[9]:


pip install python-Levenshtein


# In[10]:


from fuzzywuzzy import process

# Convert to DataFrame
df_salary = salary.copy()
df_runs = total_runs_each_year.copy()

# Function to match names
def match_names(name, names_list):
    match, score = process.extractOne(name, names_list)
    return match if score >= 80 else None  # Use a threshold score of 80

# Create a new column in df_salary with matched names from df_runs
df_salary['Matched_Player'] = df_salary['Player'].apply(lambda x: match_names(x, df_runs['Striker'].tolist()))

# Merge the DataFrames on the matched names
df_merged = pd.merge(df_salary, df_runs, left_on='Matched_Player', right_on='Striker')


# In[11]:


df_original = df_merged.copy()


# In[12]:


#susbsets data for last three years
df_merged = df_merged.loc[df_merged['Season'].isin(['2021','2022','2023'])]


# In[13]:


df_merged.Season.unique()


# In[14]:


df_merged.head()


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[16]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
X = df_merged[['runs_scored']] # Independent variable(s)
y = df_merged['Rs'] # Dependent variable
# Split the data into training and test sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a LinearRegression model
model = LinearRegression()
# Fit the model on the training data
model.fit(X_train, y_train)


# ### In a Jupyter environment, please re-run this cell to chow the HTML representation or truct the notebook.
# 
# ### On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org 

# In[17]:


X.head()


# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Assuming df_merged is already defined and contains the necessary columns
X = df_merged[['runs_scored']] # Independent variable(s)
y = df_merged['Rs'] # Dependent variable

# Split the data into training and test sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to the model (intercept)
X_train_sm = sm.add_constant(X_train)

# Create a statsmodels OLS regression model
model = sm.OLS(y_train, X_train_sm).fit()

# Get the summary of the model
summary = model.summary()
print(summary)


# In[19]:


from fuzzywuzzy import process

# Convert to DataFrame
df_salary = salary.copy()
df_runs = total_wicket_each_year.copy()

# Function to match names
def match_names(name, names_list):
    match, score = process.extractOne(name, names_list)
    return match if score >= 80 else None  # Use a threshold score of 80

# Create a new column in df_salary with matched names from df_runs
df_salary['Matched_Player'] = df_salary['Player'].apply(lambda x: match_names(x, df_runs['Bowler'].tolist()))

# Merge the DataFrames on the matched names
df_merged = pd.merge(df_salary, df_runs, left_on='Matched_Player', right_on='Bowler')


# In[20]:


df_merged[df_merged['wicket_confirmation']>10]


# In[21]:


#susbsets data for last three years
df_merged = df_merged.loc[df_merged['Season'].isin(['2021'])]


# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Assuming df_merged is already defined and contains the necessary columns
X = df_merged[['wicket_confirmation']] # Independent variable(s)
y = df_merged['Rs'] # Dependent variable

# Split the data into training and test sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to the model (intercept)
X_train_sm = sm.add_constant(X_train)

# Create a statsmodels OLS regression model
model = sm.OLS(y_train, X_train_sm).fit()

# Get the summary of the model
summary = model.summary()
print(summary)


# In[23]:


# Extract coefficients
coefficients = model.params


# In[24]:


# Construct the equation
equation = "y = " + str(round(coefficients[0], 2))
for i in range(1, len(coefficients)):
  equation += " + " + str(round(coefficients[i], 6)) + "*x" + str(i)

# Print the equation
print(equation)


# In[ ]:




