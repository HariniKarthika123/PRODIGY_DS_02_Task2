#!/usr/bin/env python
# coding: utf-8

# In[ ]:


NAME:HARINI KARTHIKA V
TASK NO:2
Prodigy InfoTech


# In[1]:


import pandas as pd

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

# Display the first few rows of the dataset
print(df.head())


# In[2]:


# Get a summary of the dataframe
print(df.info())



# In[3]:


# Check for missing values
print(df.isnull().sum())


# In[4]:


# Fill missing 'Age' values with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the mode (most common value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column because it has too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Convert 'Sex' and 'Embarked' to categorical variables
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

# Verify that there are no more missing values
print(df.isnull().sum())


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Plot the survival rate by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.show()

# Plot the distribution of ages
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Plot the survival rate by age
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Survival Rate by Age')
plt.show()

# Plot the survival rate by passenger class
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Plot the survival rate by embarkation point
plt.figure(figsize=(8, 6))
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title('Survival Rate by Embarkation Point')
plt.show()



# In[8]:


# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pair plot to explore relationships between features
sns.pairplot(df, hue='Survived', diag_kind='kde')
plt.show()


# In[ ]:




