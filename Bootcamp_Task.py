#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import os
import matplotlib.pyplot as plt


# ## Dataset description
# 
# * **Gender**: Gender of the passengers (Female, Male)
# 
# * **Customer Type**: The customer type (Loyal customer, disloyal customer)
# 
# * **Age**: The actual age of the passengers
# 
# * **Type of Travel**: Purpose of the flight of the passengers (Personal Travel, Business Travel)
# 
# * **Class**: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
# 
# * **Flight distance**: The flight distance of this journey
# 
# * **Inflight wifi service**: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)
# 
# * **Departure/Arrival time convenient**: Satisfaction level of Departure/Arrival time convenient
# 
# * **Ease of Online booking**: Satisfaction level of online booking
# 
# * **Gate location**: Satisfaction level of Gate location
# 
# * **Food and drink**: Satisfaction level of Food and drink
# 
# * **Online boarding**: Satisfaction level of online boarding
# 
# * **Seat comfort**: Satisfaction level of Seat comfort
# 
# * **Inflight entertainment**: Satisfaction level of inflight entertainment
# 
# * **On-board service**: Satisfaction level of On-board service
# 
# * **Leg room service**: Satisfaction level of Leg room service
# 
# * **Baggage handling**: Satisfaction level of baggage handling
# 
# * **Check-in service**: Satisfaction level of Check-in service
# 
# * **Inflight service**: Satisfaction level of inflight service
# 
# * **Cleanliness**: Satisfaction level of Cleanliness
# 
# * **Departure Delay in Minutes**: Minutes delayed when departure
# 
# * **Arrival Delay in Minutes**: Minutes delayed when Arrival
# 
# * **Satisfaction**: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)
# 
# *Source: kaggle.com*

# In[62]:


os.chdir('/Users/nataliapovarova/Downloads/ML_bootcamp_task')
df = pd.read_csv('train.csv')
print(df.shape)
df.head()


# In[63]:


test = pd.read_csv('test.csv')
print(test.shape)
test.head()


# ## Assignment (you can start on Dec, 10)
# 
# ### Carry out an expolaratory data analysis
# 
# 1. Check for missing values. If there are any, you should decide what to do with them:
# 
# * Usually, if most of the data (>60%) in the column is missing and the column is not crucial for modelling, you can just delete it. 
# 
# You can replace the missing data with:
# - a measure of the central tendency over the entire column
# - a measure of the central tendency within the group
# - a random element
# 
# **Quantitative data**:
# - **Continuous**
#   - Symmetric distribution:
#     - replace with **median**/mean
#   - Asymmetric distribution:
#     - replace with median
# - **Discrete**
#   - replace with mode / average
# 
# **Categorical data**:
#   - replace with mode
#   
# Another option is to create a separate model and use it for predicting missing values.
# 
# 2. Carry out univariate analysis. Use .describe(), vizualization and other methods to check out the distribution of the columns. Are there any outliers? If there are, you can drop them or replace them similarly to missing values. There are also a bunch of other methods to work with outliers, feel free to do more research!
# 
# 3. Carry out multivariate analysis. For example, you can use scatter plots and a correlation matrix. *Side note: keep in mind that correlation only checks for linear dependencies. If the correlation is small, it doesn't mean that there is no dependency at all, only that there is no **linear** dependency.*
# 
# 4. Use grouping (.group), filterings (for example, like this ``df[df[col] > df[col].quantile(.95)]``), vizualizations to formulate different hypothesis about the data. For example, maybe loyal customers are usually business travelers? Check it out! Don't forget to write down your conclusions.
# 
# The grade for this part will be based on:
# 1. Completing the plan above (2 points)
# 2. Cleanliness of your code and formatting of jupyter notebook: It should be filled with comments to your code and conclusions to your research, so we can understand follow your ideas. You should also strive to demonstrate your pandas knowledge and use as much methods from the lecture as possible (3 points)
# 3. How full your EDA is: imagine that this is a real-life project for your job. Try to test as many interesting and useful to business hypothesis as possible. You will get a higher grade for this criteria if you present a detailed and useful analysis, rather than just checking random correlations. (10 points)

# In[64]:


# your code here :)
print(df.info())
print(df.isnull().values.any()) # arrival delay in minutes has missing values


# In[11]:


non_delayed_flights = df[df['Arrival Delay in Minutes'].isnull()]


# In[12]:


non_delayed_flights['satisfaction'].describe()


# In[18]:


non_delayed_flights['satisfaction'].value_counts().plot(kind='pie', autopct='%1.1f%%');


# In[14]:


delayed_flights = df[df['Arrival Delay in Minutes'].notnull()]
delayed_flights['satisfaction'].describe()


# In[17]:


delayed_flights['satisfaction'].value_counts().plot(kind='pie', autopct='%1.1f%%');


# In[65]:


print(df['Arrival Delay in Minutes'].mode())
print(df['Arrival Delay in Minutes'].mean())
print(df['Arrival Delay in Minutes'].median())


# In[70]:


df[df['Arrival Delay in Minutes'] != 0 & df['Arrival Delay in Minutes'].notnull()]


# In[103]:


max_delay = df['Arrival Delay in Minutes'].max()
print(max_delay)
df['Arrival Delay in Minutes'].plot(kind='hist', bins=30);


# In[104]:


max_dep_delay = df['Departure Delay in Minutes'].max()
print(max_dep_delay)
df['Departure Delay in Minutes'].plot(kind='hist', bins=30);


# In[82]:


# non-delayed flights are the same as delayed flights in terms of customers' satisfaction
# so the missing values can be replaced with median (not mean, bc there is an outlier)
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())
df.info()


# In[84]:


print(df.isnull().values.any())


# In[85]:


satisfied_customers = df[df['satisfaction'] == 'satisfied']
satisfied_customers['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%');


# In[86]:


unsatisfied_customers = df[df['satisfaction'] != 'satisfied']
unsatisfied_customers['Class'].value_counts().plot(kind='pie', autopct='%1.1f%%');
# business class is the best


# In[90]:


df.groupby(['Class', 'satisfaction'])[['Age', 'Flight Distance']].agg(['min', 'max', 'mean', 'median'])
# for some reason younger people are a bit more satisfied in general
# if people use business class they are more satisfied in general
# they are also more satisfied in business class if the flight is longer and less satisfied in eco class if
# the flight is longer (which is understeandable)


# In[91]:


df.groupby(['Gender', 'satisfaction'])[['Age', 'Flight Distance']].agg(['min', 'max', 'mean', 'median'])
# gender doesn't have any significant impact


# In[94]:


business_class = df[df['Class'] == 'Business']
business_class['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%');


# In[95]:


satisfied_customers['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%');


# In[92]:


df.groupby(['Customer Type', 'satisfaction'])[['Age', 'Flight Distance']].agg(['min', 'max', 'mean', 'median'])


# In[96]:


# loyal customers are predictably more satisfied
satisfied_customers['Customer Type'].value_counts().plot(kind='pie', autopct='%1.1f%%');


# In[93]:


df.groupby(['Type of Travel', 'satisfaction'])[['Age', 'Flight Distance']].agg(['min', 'max', 'mean', 'median'])


# In[97]:


# business travellers are also more satisfied, and they are more satisfied if the flight is longer
satisfied_customers['Type of Travel'].value_counts().plot(kind='pie', autopct='%1.1f%%');


# In[98]:


# apparently, business class passengers are more satisfied, because their companies pay for their tickets
# they are also more satisfied in longer flights
business_class['Type of Travel'].value_counts().plot(kind='pie', autopct='%1.1f%%');


# In[100]:


# food and seat comfort makes difference
df.groupby(['Type of Travel', 'Class', 'satisfaction'])[['Food and drink', 'Seat comfort']].agg(['min', 'max', 'mean', 'median'])


# In[101]:


df.groupby(['Class', 'satisfaction'])[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].agg(['min', 'max', 'mean', 'median'])

# obviously people are more satisfied when the delay is smaller
# but delays have outliers, so mean number is not very robust here


# In[108]:


# dealing with outliers
int_float_columns = []
for column in df.columns:
    if (df[column].dtypes == 'int64') or (df[column].dtypes == 'float64'):
        int_float_columns.append(column)
print(len(int_float_columns))
#boxplot = df.boxplot(column=int_float_columns)


# In[113]:


boxplot_1 = df.boxplot(column='Age')


# In[120]:


# delays are left out, because it is already known that there are outliers
boxplot_2 = df.boxplot(column=int_float_columns[3:8])


# In[117]:


boxplot_3 = df.boxplot(column=int_float_columns[8:13])


# In[121]:


boxplot_4 = df.boxplot(column=int_float_columns[13:17])


# In[124]:


print(df['Checkin service'].mean())
print(df['Checkin service'].median())


# In[127]:


print(set(df['Checkin service'].tolist()))
df['Checkin service'].value_counts().plot(kind='pie', autopct='%1.1f%%');
# won't count checkin service values as outliers


# In[129]:


len(df[df['Departure Delay in Minutes'] == df['Departure Delay in Minutes'].max()])


# In[131]:


len(df[df['Arrival Delay in Minutes'] == df['Arrival Delay in Minutes'].max()])


# In[135]:


(df['Arrival Delay in Minutes'].describe())


# In[136]:


df['Arrival Delay in Minutes'].median()


# In[137]:


df['Arrival Delay in Minutes'].mode()


# In[139]:


dep_outlier_deleted = df[df['Departure Delay in Minutes'] != df['Departure Delay in Minutes'].max()]
dep_outlier_deleted['Departure Delay in Minutes'].plot(kind='hist', bins=30);


# In[140]:


print(set(df['Departure Delay in Minutes'].tolist()))


# ## Assignment (you can start on Dec, 11)
# 
# 1. Create new features based on your EDA. Don't forget to check how they performed after you are finished with modelling! You can use ``feature_importances`` from scikit-learn or use advanced methods like SHAP or Lime. (5 points)
# 2. Your target variable is ``satisfaction``. You should research metrics and choose one or multiple that you will use to validate your model. Write down the formula(s) and your motivation to use them. (3 points)
# 2. Design the validation process: for example, will you use cross-validation or just train-test split? Will you account for the imbalance in classes, if it exists? (2 points)
# 3. Experiment with modelling. You can use models from the lecture or do your own research. You can also try out approaches like stacking and blending – will they increase the quality? (15 points)
# 4. Make predictions on the test.csv dataset.

# In[ ]:


# your code here :)

