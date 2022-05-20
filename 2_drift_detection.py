# %%
from re import X
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date, datetime
import os.path
from sklearn.model_selection import train_test_split

# %%
previous_r2 = 0.372
previous_rmse = 0.329

# %%
model = pickle.load(open("mlruns/0/1fcae12c009b4817b459d7caf566030c/artifacts/model/model.pkl", 'rb'))
test_data = pd.read_csv("data/weatherAUS_clean.csv")

# %%
test_data.drop(test_data.columns[[0]], axis=1, inplace=True)

# %%
test_data

# %%
test_data.loc[:,'MinTemp'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)
test_data.loc[:,'MaxTemp'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)
test_data.loc[:,'Rainfall'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)
test_data.loc[:,'Evaporation'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)
test_data.loc[:,'Sunshine'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)
test_data.loc[:,'WindGustSpeed'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)

# %%
test_data

# %%
X = test_data.iloc[:,:-1]
y = test_data.iloc[:,-1]

# %%
predictions = model.predict(X)

# %%
RMSE = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)
print('RMSE on test data: ', RMSE)
print('r2 on test data: ', r2)

# %%
### Hard test ###

hard_test_RMSE = previous_r2 > np.mean(RMSE)
hard_test_r2 = previous_r2 < np.mean(r2)
print('\nLegend: \nTRUE means the model has drifted. FALSE means the model has not.')
print('\n.. Hard test ..')
print('RMSE: ', hard_test_RMSE, '  R2: ', hard_test_r2)

# %%
### Parametric test ###
param_test_RMSE = previous_r2 > np.mean(RMSE) + 2*np.std(RMSE)
param_test_r2 = previous_r2 < np.mean(r2) - 2*np.std(r2)

print('\n.. Parametric test ..')
print('RMSE: ', param_test_RMSE, '  R2: ', param_test_r2)

# %%
### Non-parametric (IQR) test ###
iqr_RMSE = np.quantile(RMSE, 0.75) - np.quantile(RMSE, 0.25)
iqr_test_RMSE = previous_r2 > np.quantile(RMSE, 0.75) + iqr_RMSE*1.5

iqr_r2 = np.quantile(r2, 0.75) - np.quantile(r2, 0.25)
iqr_test_r2 = previous_r2 < np.quantile(r2, 0.25) - iqr_r2*1.5

print('\n.. IQR test ..')
print('RMSE: ', iqr_test_RMSE, '  R2: ', iqr_test_r2)

# Re-training signal
drift_df = pd.DataFrame()
drift_signal_file = 'evaluation/model_drift.csv'
now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

# %%
print('\n  --- DRIFT DETECTION ---')

actual_tests = {
  'hard_test_RMSE': hard_test_RMSE,
  'hard_test_r2': hard_test_r2,
  'param_test_RMSE': param_test_RMSE,
  'param_test_r2': param_test_r2,
  'iqr_test_RMSE': iqr_test_RMSE,
  'iqr_test_r2': iqr_test_r2
}

# %%
actual_tests

# %%
a_set = set(actual_tests.values())
a_set

# %%
if True in set(actual_tests.values()):
  print("")

a_set = set(actual_tests.values())
drift_detected = True in set(actual_tests.values())

if drift_detected:
  print("Drift Detected!")
