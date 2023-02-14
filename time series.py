#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Python libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import stats
from math import pi, sqrt, ceil
from warnings import catch_warnings
from warnings import filterwarnings
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot


# In[2]:


# Import Time Series libraries
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[3]:


# Read raw data
df = pd.read_excel(r"C:\Users\thejalraj\OneDrive\Desktop\Project\Ak data.xlsx")
df


# In[4]:


# Create pretty x axis labels
def get_x_labels(all_labels):
    x_labels = []
    for ix in range(len(all_labels)):
        if ix % 5 == 0:
            date_label = str(all_labels[ix]).replace('T', '*').split('*')[0]
            x_labels.append(date_label)
        else:
            x_labels.append('')
    return x_labels


# In[5]:


# Cooking weekly data
x_date = np.array(df.index)
x_data = np.array(range(len(x_date))).reshape((-1, 1))
y_data = np.array(df.order_quantity)
xticks = get_x_labels(x_date)
y_max = max(y_data) * 1.01


# In[6]:


# Grouping data by year
df.groupby(['year'])['sales'].agg('sum')


# In[7]:


# Cooking data for year 2018
data_2018 = df[df['year'] == 2018]['order_quantity']
print('2018 Records: %s' % len(data_2018))
print('Total cases: %s' % sum(data_2018))
print('Average cases: %.2f' % np.mean(data_2018))
print('Median cases: %.2f' % np.median(data_2018))
print('Standard deviation: %.2f' % np.std(data_2018))


# In[8]:


# Plot trends by year
plt.figure(figsize=(18, 7), dpi=200)
for i in range(2015, 2023):
    ts_data = df[df['year'] == i]['sales']
    plt.plot(range(len(ts_data)), ts_data, label=i)
plt.ylim((0, y_max))
plt.title('Annual Sale trends', fontsize=16)
plt.xlabel('Month', fontsize=10)
plt.ylabel('sales', fontsize=10)
plt.xticks(fontsize=10)
plt.legend(loc='lower right')
plt.show()


# In[9]:


# Plot trends by year
plt.figure(figsize=(18, 7), dpi=200)
for i in range(2015, 2023):
    ts_data = df[df['year'] == i]['order_quantity']
    plt.plot(range(len(ts_data)), ts_data, label=i)
plt.ylim((0, y_max))
plt.title('Annual Order trends', fontsize=16)
plt.xlabel('Month', fontsize=10)
plt.ylabel('order_quantity', fontsize=10)
plt.xticks(fontsize=10)
plt.legend(loc='lower right')
plt.show()


# In[10]:


# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi= 200)
sns.boxplot(x='year', y='sales', data=df, ax=axes[0])
sns.boxplot(x='Month', y='sales', data=df, ax=axes[1])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=16); 
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=16)
plt.show()


# In[11]:


# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi= 200)
sns.boxplot(x='year', y='order_quantity', data=df, ax=axes[0])
sns.boxplot(x='Month', y='order_quantity', data=df, ax=axes[1])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=16); 
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=16)
plt.show()


# In[12]:


# Calculate root mean squared error or RMSE
def calc_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# In[13]:


# Calculate mean absolute percentage error or MAPE
def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def calc_mape(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


# In[14]:


# Calculate confidence interval
def get_interval(y, y_pred, pi=0.99):
    n = len(y)
    
    # Get standard deviation of y_test
    sum_errs = np.sum((y - y_pred)**2) / (n - 2)
    stdev = np.sqrt(sum_errs)
    
    # Get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
    
    return interval


# In[15]:


#LINEAR REG


# In[16]:


# Create model LR
degree = 4
x_ = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(x_data)
model = LinearRegression().fit(x_, y_data)

# Validate model
r_sq = model.score(x_, y_data)
print('Correlation:', sqrt(r_sq))
print('Coefficient of determination:', r_sq)
print('Intercept:', model.intercept_, ', Slope:', model.coef_)


# In[17]:


# Prediction
y_pred = model.predict(x_)

# Calculate errors
rmse = calc_rmse(y_data, y_pred)
mape = calc_mape(y_data, y_pred)
print('The RMSE of our forecasts is: {}'.format(round(rmse, 3)))
print('The MAPE of our forecasts is: {} %'.format(round(mape, 3)))


# In[18]:


ci_alpha = 0.9
ci = get_interval(y_data, y_pred, ci_alpha)
print('Conf. Int.:', ci)


# In[19]:


# Plot chart
plt.figure(figsize=(18, 7), dpi=200)
plt.plot(x_date, y_data, '-', label='sales')
plt.plot(x_date, y_pred, '-', color='green', label='Trend')
plt.plot(x_date, y_pred + ci, '--', color='darkgreen', label='CI High')
plt.plot(x_date, y_pred - ci, '--', color='darkgreen', label='CI Low')
plt.ylim((0, y_max))
plt.title('Monthly sale with Trends', fontsize=16)
plt.xlabel('year', fontsize=10)
plt.ylabel('sales', fontsize=10)
plt.xticks(fontsize=10, rotation=45)
plt.legend()
plt.show()


# In[20]:


test = df.tail(12)


# In[40]:


train = df.head(84)


# In[22]:


tsa_plots.plot_acf(df.sales, lags = 12)


# In[23]:


model = ARIMA(train.sales, order = (12,1,6))
res = model.fit()


# In[24]:


print(res.summary())


# In[25]:


start = len(train)
end = start + len(test)-1
pred = res.predict(start = start, end = end)
pred


# In[26]:


# Evaluate forecasts
rmse = sqrt(mean_squared_error(test.sales, pred))
print('Test RMSE: %.3f' % rmse)


# In[27]:


pyplot.plot(test.sales, color ='blue')
pyplot.plot(pred, color='red')
pyplot.show()


# In[28]:


get_ipython().system('pip install pmdarima')
import pmdarima as pm


# In[29]:


#auto_ar_model = pm.auto_arima(train.Sales, start_p = 0, start_q = 0, max_p = 12, max_q = 12, m = 1, d=None, seasonal=False,
                             # start_P=0, trace=True, error_action='warn',stepwise=True)
auto_model = pm.auto_arima(df['sales'], trace=True, suppress_warnings=True)


# In[30]:


# Best Parameters ARIMA
# ARIMA with AR=0, I = 1, MA = 0
model = ARIMA(train.sales, order = (2,1,2))
res = model.fit()
print(res.summary())


# In[31]:


start = len(train)
end = start + len(test)-1
pred = res.predict(start = start, end = end)
pred


# In[32]:


pyplot.plot(test.sales, color ='blue')
pyplot.plot(pred, color='red')
pyplot.show()


# In[33]:


rmse = sqrt(mean_squared_error(test.sales, pred))
print('Test RMSE: %.3f' % rmse)


# In[34]:


test["sales"].mean()


# In[35]:


model2 = ARIMA(df.sales, order = (2,1,2))
model2 = model2.fit()
df.tail()


# In[36]:


pred = model2.predict(start = len(df), end = len(df)+24, typ = 'levels').rename("Arima Prediction")
pred


# In[37]:


pyplot.plot(df.sales, color ='red')
pred.plot(figsize = (12,5), legend = True)


# In[38]:


tsa_plots.plot_acf(pred, lags = 12)


# In[ ]:




