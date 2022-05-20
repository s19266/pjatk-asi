# %%
from tabnanny import verbose
import numpy as np
import pandas as pd
import os
from urllib.parse import urlparse
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

import logging

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

csv_url = ("./data/weatherAUS.csv")
try:
  dataset = pd.read_csv(csv_url, sep=',')
except Exception as e:
  logger.exception(f'Oj, coś poszło nie tak. Error: {e}')

# %%
dataset.head(10)
dataset.shape

# %%
dataset.RainToday

# %%
dataset['RainTomorrow'] = dataset['RainTomorrow'].map({'Yes': 1, 'No': 0})
dataset['RainToday'] = dataset['RainToday'].map({'Yes': 1, 'No': 0})

# %%
# Checking percentage of missing data in every column
(dataset.isnull().sum()/len(dataset))*100

# %%
# Filling the missing values for continuous variables with mode
dataset['MinTemp']=dataset['MinTemp'].fillna(dataset['MinTemp'].mean())
dataset['MaxTemp']=dataset['MinTemp'].fillna(dataset['MaxTemp'].mean())
dataset['Rainfall']=dataset['Rainfall'].fillna(dataset['Rainfall'].mean())
dataset['Evaporation']=dataset['Evaporation'].fillna(dataset['Evaporation'].mean())
dataset['Sunshine']=dataset['Sunshine'].fillna(dataset['Sunshine'].mean())
dataset['WindGustSpeed']=dataset['WindGustSpeed'].fillna(dataset['WindGustSpeed'].mean())
dataset['WindSpeed9am']=dataset['WindSpeed9am'].fillna(dataset['WindSpeed9am'].mean())
dataset['WindSpeed3pm']=dataset['WindSpeed3pm'].fillna(dataset['WindSpeed3pm'].mean())
dataset['Humidity9am']=dataset['Humidity9am'].fillna(dataset['Humidity9am'].mean())
dataset['Humidity3pm']=dataset['Humidity3pm'].fillna(dataset['Humidity3pm'].mean())
dataset['Pressure9am']=dataset['Pressure9am'].fillna(dataset['Pressure9am'].mean())
dataset['Pressure3pm']=dataset['Pressure3pm'].fillna(dataset['Pressure3pm'].mean())
dataset['Cloud9am']=dataset['Cloud9am'].fillna(dataset['Cloud9am'].mean())
dataset['Cloud3pm']=dataset['Cloud3pm'].fillna(dataset['Cloud3pm'].mean())
dataset['Temp9am']=dataset['Temp9am'].fillna(dataset['Temp9am'].mean())
dataset['Temp3pm']=dataset['Temp3pm'].fillna(dataset['Temp3pm'].mean())
dataset['RainToday']=dataset['RainToday'].fillna(dataset['RainToday'].mode()[0])
dataset['RainTomorrow']=dataset['RainTomorrow'].fillna(dataset['RainTomorrow'].mode()[0])
dataset['WindDir9am'] = dataset['WindDir9am'].fillna(dataset['WindDir9am'].mode()[0])
dataset['WindGustDir'] = dataset['WindGustDir'].fillna(dataset['WindGustDir'].mode()[0])
dataset['WindDir3pm'] = dataset['WindDir3pm'].fillna(dataset['WindDir3pm'].mode()[0])

# %%
# Encoding the categorical variables
le = LabelEncoder()
dataset['Location'] = le.fit_transform(dataset['Location'])
dataset['WindDir9am'] = le.fit_transform(dataset['WindDir9am'])
dataset['WindDir3pm'] = le.fit_transform(dataset['WindDir3pm'])
dataset['WindGustDir'] = le.fit_transform(dataset['WindGustDir'])

# %%
dataset

# %%
# Drop highly correlated columns
dataset = dataset.drop(['Temp3pm', 'Temp9am', 'Humidity9am'],axis=1)

# %%
dataset.columns

# %%
# Drop date column
dataset=dataset.drop(['Date'],axis=1)
dataset

# %%
dataset
dataset.to_csv('data/weatherAUS_clean.csv')

# %%
# TRAIN
x_train, x_test, y_train, y_test = train_test_split(
  dataset.iloc[:,:-1],
  dataset.iloc[:,-1],
  test_size=0.2,
  random_state=42
)

# %%
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def log_output(model_name, model, actual, pred):
  (rmse, mae, r2) = eval_metrics(actual, pred)

  print(model_name)
  print(f"  RMSE    : {rmse}")
  print(f"  MEA     : {mae}")
  print(f"  R2      : {r2}")

  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)

  tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

  # Model registry does not work with file store
  if tracking_url_type_store != "file":
    mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
  else:
    mlflow.sklearn.log_model(model, "model")

# %%
mlflow.start_run()

# %%
# LogisticRegression
with mlflow.start_run(run_name="Logistic Regression", nested=True):
  model = LogisticRegression(max_iter=500, solver='liblinear')
  model.fit(x_train, y_train)
  predicted = model.predict(x_test)

  log_output("LogisticRegression", model, y_test, predicted)

# %%
# Gaussian Naive Bayes
with mlflow.start_run(run_name="Gaussian Naive Bayes", nested=True):
  model = GaussianNB()
  model.fit(x_train, y_train)

  predicted = model.predict(x_test)

  log_output("GaussianNB", model, y_test, predicted)

# %%
# Bernoulli Naive Bayes
with mlflow.start_run(run_name="Bernoulli Naive Bayes", nested=True):
  model = BernoulliNB()
  model.fit(x_train, y_train)

  predicted = model.predict(x_test)

  log_output("BernoulliNB", model, y_test, predicted)

# %%
# Random Forest
with mlflow.start_run(run_name="RandomForest", nested=True):
  model = RandomForestRegressor(n_estimators = 100, random_state = 0)
  model.fit(x_train, y_train)

  predicted = model.predict(x_test)

  log_output("RandomForestRegressor", model, y_test, predicted)

# %%
mlflow.end_run()

# %%
