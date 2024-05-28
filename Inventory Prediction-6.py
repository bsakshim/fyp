#!/usr/bin/env python
# coding: utf-8
# In[2]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime
from displayfunction import display
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae

import warnings

warnings.filterwarnings('ignore')

# In[3]:

df = pd.read_csv('StoreDemand.csv')
display(df.head())
display(df.tail())

# In[4]:


df.shape

# In[5]:


df.info()

# In[6]:


df.describe()

# In[7]:


parts = df["date"].str.split("-", n=3, expand=True)
df["year"] = parts[0].astype('int')
df["month"] = parts[1].astype('int')
df["day"] = parts[2].astype('int')
df.head()

# In[9]:


#from datetime import datetime
import calendar


def weekend_or_weekday(year, month, day):
    d = datetime(year, month, day)
    if d.weekday() > 4:
        return 1
    else:
        return 0


df['weekend'] = df.apply(lambda x: weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)
df.head()

# In[10]:


df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))
df.head()


# In[11]:


def which_day(year, month, day):
    d = datetime(year, month, day)
    return d.weekday()


df['weekday'] = df.apply(lambda x: which_day(x['year'],
                                             x['month'],
                                             x['day']),
                         axis=1)
df.head()

# In[12]:


df.drop('date', axis=1, inplace=True)

# In[13]:


df['store'].nunique(), df['item'].nunique()

# In[14]:


features = ['store', 'year', 'month',
            'weekday', 'weekend', ]

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    df.groupby(col).mean()['sales'].plot.bar()
plt.show()

# In[15]:


plt.figure(figsize=(10, 5))
df.groupby('day').mean()['sales'].plot()
plt.show()

# In[16]:


plt.figure(figsize=(15, 10))

# Calculating Simple Moving Average 
# for a window period of 30 days 
window_size = 30
data = df[df['year'] == 2013]
windows = data['sales'].rolling(window_size)
sma = windows.mean()
sma = sma[window_size - 1:]

data['sales'].plot()
sma.plot()
plt.legend()
plt.show()

# In[17]:


plt.subplots(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.distplot(df['sales'])

plt.subplot(1, 2, 2)
sb.boxplot(df['sales'])
plt.show()

# In[18]:


plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.8,
           annot=True,
           cbar=False)
plt.show()

# In[19]:


df = df[df['sales'] < 140]

# In[20]:


features = df.drop(['sales', 'year'], axis=1)
target = df['sales'].values

X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                  test_size=0.05,
                                                  random_state=22)
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

# In[21]:


# Normalizing the features for stable and fast training. 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# In[22]:


models = [LinearRegression(), XGBRegressor(), Lasso(), Ridge()]

for i in range(4):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')

    train_preds = models[i].predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))

    val_preds = models[i].predict(X_val)
    print('Validation Error : ', mae(Y_val, val_preds))
    print(val_preds)
for i in range(4):
    kk = models[i].fit(X_train, Y_train)
    print(kk)

    print(f'{models[i]}: ')

    train_preds = models[i].predict(X_train)
    print('Training Error:', mae(Y_train, train_preds))

# In[23]:


# Print the shapes of the training and testing sets
print("Training set shape:")
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("\nTesting set shape:")
print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)

# In[24]:


print(len(val_preds))

# In[25]:


val_preds.shape

# In[43]:


#from xgboost import XGBRegressor
#from sklearn.datasets import make_regression
#from sklearn.model_selection import train_test_split

# Generate some example regression data
X, Y = make_regression(n_samples=1000, n_features=8, noise=0.1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the model
model = XGBRegressor()
#model = LinearRegression()
model.fit(X_train, Y_train)
# Save the trained model
#model.save_model('your_model.json')

# In[72]:


# Assuming `new_data` is your new dataset
# new_data = pd.read_csv('StoreDemand.csv')
new_data = X_test
# Preprocess new_data as necessary...
# Make predictions
predictions = model.predict([[1, 3, 6, 19, 1, 0, 0, 5]])
print(predictions)

# In[73]:
print(predictions.shape)
print(features)
# In[ ]:
