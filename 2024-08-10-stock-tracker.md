## **Online Sales Prediction in Canada**

### **Contents**
- [Data](#Data)
- [Single-Category ML Model](#Single-Category-ML-Model)
- [Categorical Model](#Categorical-Model)
- [Model Evaluation](#Model-Evaluation)

## **Specific Subject**
Based on the available dataset:
- Food  
- Sporting and athletic goods  

## **Objective**
- Develop a web-based application to forecast product sales using time-series analysis.
- Identify buyer habit trends.  
- Estimate potential profit (average predicted sales).  
- Provide insights to refine sales strategies and improve customer retention.  
- Assess product success or failure.

## **Data**  
- Gather structured sales data through the Canada government API.
- Perform exploratory data analysis (EDA) and feature engineering.  
- Clean and preprocess tabulated sales data.  
- Integrate time-based features for trend analysis.  
- Ingest data using Spark for efficient handling of large datasets.  
- Validate data completeness, uniqueness, and compliance with constraints.  

## **Single-Category ML Model**  
- Train and benchmark ML-based algorithms (e.g., Random Forest, Decision Tree).  
- Optimize model performance using GPU acceleration.  
- Apply cross-validation techniques.  
- Evaluate model accuracy using test datasets.  

## **Categorical Model**  
- Develop a multi-category sales forecasting model.  
- Compare model performance across different product categories.  
- Use feature selection to improve predictive accuracy.  

## **Model Evaluation**  
- Compare accuracy metrics (MAE, RMSE, R²) across models.  
- Validate performance using test datasets.  
- Identify areas for improvement and fine-tune hyperparameters.  

## **Data Visualizations**  
- Map consumer purchase trends.  
- Display correlation between product profitability and consumer demographics.  
- Illustrate the impact of external factors (e.g., economic shifts, seasonal trends, pandemic-related restrictions).  

## **Model Presentation and Results**  
- Deploy the predictive model on an interactive website using Flask.  
- Provide an intuitive interface for businesses to input product details and receive sales forecasts.  

## Import libraries
```python
# Libraries required to run this notebook
# ============
import os, sys
import seaborn as sns
import requests
from requests_oauthlib import OAuth1
import json 
import simplejson as json
from bs4 import BeautifulSoup
from retrying import retry
from urllib.parse import quote, quote_plus
from urllib.request import urlopen
from datetime import datetime
from ediblepickle import checkpoint
from time import sleep
import pandas as pd
import re
import numpy as np
import warnings
import matplotlib as plt
import pickle
import dill
import altair as alt

# Scikit-learn 
# ============
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, accuracy_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn import base

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
%matplotlib inline
```

## Data: Uploading Data

Loading preprocessed data with additional features from saved files (generated in the stocktracker_cats_preprocessing notebook). Ultimately, data ingestion and cleaning should be automated using data pipelines. The feature engineering process and machine learning model development should be performed using data sourced from a centralized data warehouse.

```python
We want to build a ML model that can predict inventory for all the following product categories: 
***************
1. elliptical
2. dumbells
3. indoor bike
4. weight lift
5. treadmill
***************

hist_data = pd.read_json('hist_data.json')
true_data = pd.read_json('true_df.json')
```

## Data preparation: Adding new features

```python
hist_data["Year"] = hist_data.date.dt.year
hist_data["Month"] = hist_data.date.dt.month
hist_data["Week"] = hist_data.date.dt.isocalendar().week
hist_data["Weekday"] = hist_data.date.dt.weekday
hist_data["Day"] = hist_data.date.dt.day
hist_data["Dayofyear"] = hist_data.date.dt.dayofyear
hist_data["Quarter"] = hist_data.date.dt.quarter

df_model = hist_data.drop(columns=['date', 'title', 'sku', 'NewReleaseFlag'])
```


## Single-ML

First build and optimize a ML model for one category. This model then will be extended for multi-category cases in next section.

```python

# first try out the simple model for individual 
df_model = df_model.loc[df_model.product_category == 'dumbells'].drop(columns=['product_category'])

# Train-test split
y = df_model['inventory']
X = df_model.drop(columns=['inventory', 'Weekday','Dayofyear'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=0)


# Model
min_tree_splits = range(2,6)
min_tree_leaves = range(2,6)
nmax_features = range(1, 11)#range(0,20)
max_tree_depth = range(1, 8)#range(0,10)


categorical_columns = ['Year','Month', 'Week', 'Day','Quarter']
numeric_columns = ['price', 'ReleaseNumber']
trans_columns = ColumnTransformer([
    ('numeric', 'passthrough', numeric_columns),
    ('categorical','passthrough' , categorical_columns)

])

features = Pipeline([
    ('columns', trans_columns),
    ('scaler', StandardScaler()),        
])

param_grid = {
    'min_samples_split': min_tree_splits,
    'min_samples_leaf': min_tree_leaves,
    'max_depth': max_tree_depth,
    'max_features':nmax_features
}

gs = GridSearchCV(
                    DecisionTreeRegressor(min_samples_split=2, min_samples_leaf=6, max_depth=9), 
                    param_grid, cv=3, n_jobs=2
                 )

pipe = Pipeline([('feature', features), ('gs_est', gs)])


pipe.fit(X_train, y_train);

# calculate error 
def rmse(model, X, y):
    return np.sqrt(metrics.mean_squared_error(model.predict(X), y))

# and for mean model
err_mean = np.sqrt(np.array(y_test).flatten().var())

# get best_estimator paramters
gcv = pipe.named_steps['gs_est']
best_regress = gcv.best_estimator_
best_regress.get_params()

y_pred = pipe.predict(X_test)

print("*" * 50)
# Find the error
print("RMSE: {} & err_mean: {}".format(rmse(pipe, X_test, y_test), err_mean))

# Compute metrics
print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
R2score =  r2_score(y_test, y_pred)
print("R2 score is {}".format(R2score))
print("*" * 50)

import matplotlib.pyplot as plt
features = X.columns#df_model.drop(columns=['inventory']).coluns
importances = best_regress.feature_importances_
indices = np.argsort(importances)

importance = list(importances)
feat_importance = pd.DataFrame(importance, features).reset_index()
feat_importance.columns=['feature', 'importance']
feat_importance.to_json('feature_importance.json')

import altair as alt
fimp = pd.read_json('feature_importance.json')
chart = alt.Chart(fimp
                ).mark_bar(  
                   opacity = 0.8,
                    strokeWidth=1
                ).encode(
                x=alt.X("importance:Q", axis=alt.Axis(title='')),
                y=alt.Y("feature:N", axis=alt.Axis(title='')),
                color=alt.value("blue"), 
        ).properties(title="Feature importance",width=800,height=200
        )

combined = chart

chart

**************************************************
RMSE: 196.89467885595099 & err_mean: 315.8465531521096
MAE: 125.91135845954335
MSE: 38767.51456178807
R2 score is 0.6113884760210656
**************************************************
```

<div align="center">
  <img src="/assets/images/blogs/stock_tr_ML.png" width="1200px" height="300">
</div>

## Categorical-ML

```python
class GroupbyEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, column, estimator_factory):
        # column is the value to group by; estimator_factory can be
        # called to produce estimators
        self.column = column
        self.est_fact = estimator_factory
        self.est_dict = {}
    
    def fit(self, X, y):
        X = X.copy()
        X['label'] = y
        # Create an estimator and fit it with the portion in each group
        for key, df_pcat in X.groupby(self.column):
            self.est_dict[key] = self.est_fact().fit(df_pcat, df_pcat['label'])
        return self

    def predict(self, X):
        X = X.copy()
        X['label'] = 0
        predict_dic = {}
        cats = X[self.column].unique().tolist()        

        for key, df_pcat in X.groupby(self.column):
            predict_dic[key] = self.est_dict[key].predict(df_pcat)
                                
        ordered_predict_list = [predict_dic[k] for k in cats]
        return np.concatenate(ordered_predict_list)

def category_factory():
    
    min_tree_splits = range(2,6)
    min_tree_leaves = range(2,8)
    nmax_features = range(0,10)
    max_tree_depth = range(0,20)

    # categorical_columns = ['Quarter','Month', 'Week', 'Dayofyear', 'Day']
    categorical_columns = ['Year','Month', 'Week', 'Day','Quarter']
    numeric_columns = ['price', 'ReleaseNumber']
    trans_columns = ColumnTransformer([
        ('numeric', 'passthrough', numeric_columns),
        ('categorical','passthrough' , categorical_columns)

    ])

    features = Pipeline([
        ('columns', trans_columns),
        ('scaler', MaxAbsScaler()),
    ])
    

    param_grid = {
                  'max_depth' : max_tree_depth,
                  'max_features':nmax_features,
                  'min_samples_leaf':min_tree_leaves 
                 }

    gs = GridSearchCV(
                        DecisionTreeRegressor(min_samples_split=2), 
                        param_grid, cv=40, n_jobs=2
                     )

    
    pipe = Pipeline([('feature', features), ('gs_est', gs)])
    
    return pipe
```

```python
categoty_model =  GroupbyEstimator('product_category', category_factory).fit(df_model, df_model['inventory'])
print('R^2 score using selected columns and transformers: {} '.format(categoty_model.score(df_model, df_model['inventory'])))

R^2 score using selected columns and transformers: 0.9178315823491655 
```

## Model Evaluation

```python
# Pickle the estimator
with open('cat_model.pkl', 'wb') as file:
    pickle.dump(categoty_model, file)

with open('cat_model.dill', 'wb') as f:
    dill.dump(categoty_model, f, recurse=True)

hist_data.to_json('hist_dataset.json')
true_data.to_json('true_dataset.json')

with open('cat_model.dill', 'rb') as f:
    cat_model = dill.load(f)

# Load historical and test data
hist_data = pd.read_json('hist_dataset.json')
true_data = pd.read_json('true_dataset.json')

# what category to predict
cat_to_pred = 'treadmill' 

# now clean test data, add features and make predictions
# create a test dataFrame
true_df = true_data.drop(columns=['title', 'sku', 'NewReleaseFlag'])

# select corresponding collumns
true_df = true_df.loc[true_df.product_category == cat_to_pred]

true_df["Year"] = true_df.date.dt.year
true_df["Month"] = true_df.date.dt.month
true_df["Week"] = true_df.date.dt.isocalendar().week
true_df["Weekday"] = true_df.date.dt.weekday
true_df["Day"] = true_df.date.dt.day
true_df["Dayofyear"] = true_df.date.dt.dayofyear
true_df["Quarter"] = true_df.date.dt.quarter

true_df = true_df.drop(columns=['date'])

# create a y_test vector
y_test = true_df['inventory']

y_pred = cat_model.predict(true_df)

print("*" * 50)
# Find the error
# print("RMSE: {} & err_mean: {}".format(rmse(pipe, X_test, y_test), err_mean))
# Compute metrics
print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
R2score =  r2_score(y_test, y_pred)
print("R2 score is {}".format(R2score))
print("*" * 50)

**************************************************
MAE: 32.521043802879504
MSE: 2748.320011520275
R2 score is 0.8233724604015465
**************************************************

```

```python
#Plotting the results

dh = hist_data.loc[hist_data.product_category == cat_to_pred]#pd.read_json('treadmill_hist_data.json')
dh_gr = pd.DataFrame(dh.groupby('date').inventory.sum().reset_index())
dt = true_data.loc[true_data.product_category == cat_to_pred]
dt_gr = pd.DataFrame(dt.groupby('date').inventory.sum().reset_index())

# now create a new df where the columnlabel is replaced by the predicted label
dp = dt
dp['inventory'] = y_pred
dp_gr = pd.DataFrame(dp.groupby('date').inventory.sum().reset_index())


chart1 = alt.Chart(dh_gr
                ).mark_line(  
                   opacity = 0.8,
                    strokeWidth=2
                ).encode(
                x=alt.X("date:T", axis=alt.Axis(title='')),
                y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
                color=alt.value("blue"), 
        ).properties(title="Historical/validation data for {}".format(cat_to_pred),width=800,height=200
        )

chart2 = alt.Chart(dt_gr
                ).mark_line(  
                   opacity = 0.8,
                    strokeWidth=2
                ).encode(
                x=alt.X("date:T", axis=alt.Axis(title='')),
                y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
                color=alt.value("red"), 
        ).properties(title="Historical/validation data for {}".format(cat_to_pred),width=800,height=200
        )

combined1 = chart1 + chart2 


dh_gr = dh_gr[dh_gr.date >= pd.to_datetime('2021-05-01')]
dt_gr = dt_gr[dt_gr.date >= pd.to_datetime('2021-05-01')]
dp_gr = dp_gr[dp_gr.date >= pd.to_datetime('2021-05-01')]


chart3 = alt.Chart(dh_gr
                ).mark_line(  
                   opacity = 0.8,
                    strokeWidth=2
                ).encode(
                x=alt.X("date:T", axis=alt.Axis(title='')),
                y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
                color=alt.value("blue"), 
        ).properties(title="",width=800,height=200
        )

chart4 = alt.Chart(dt_gr
                ).mark_line(  
                   opacity = 0.8,
                    strokeWidth=2
                ).encode(
                x=alt.X("date:T", axis=alt.Axis(title='')),
                y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
                color=alt.value("red"), 
        ).properties(title="",width=800,height=200
        )

chart5 = alt.Chart(dp_gr
                ).mark_line(  
                   opacity = 0.8,
                    strokeWidth=2
                ).encode(
                x=alt.X("date:T", axis=alt.Axis(title='')),
                y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
                color=alt.value("darkorange"), 
        ).properties(title="",width=800,height=200
        )

combined2 = chart3 + chart4 + chart5

combined1 & combined2

```
<div align="center">
  <img src="/assets/images/blogs/stock_tr_catML.png" width="700px" height="400">
</div>
