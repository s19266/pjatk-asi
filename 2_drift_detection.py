from re import X
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from sklearn.metrics import accuracy_score
from pathlib import Path

from utils.train import train
from utils.prepare import prepare
from os import environ, path
from colorama import Fore, Style

mlflow.set_tracking_uri(environ.get('TRACKING_URI', "http://localhost:5001"))
data_path = 'data/batches'
filenames = list(map(
  lambda child: child.name,
  filter(lambda child: child.is_file(), Path(data_path).glob('*.csv'))
))

# Load latest model
model = mlflow.pyfunc.load_model(model_uri=f"models:/Best/latest")

# Get mean of all previous accuracies
r = mlflow.search_runs(filter_string="tags.mlflow.runName = 'Best' AND attribute.status = 'FINISHED'")
previous_accuracy = r[['metrics.accuracy']].mean()

last_batch_number = 0

for filename in filenames:
  (number_as_string, extension) = filename.split('.')
  number = int(number_as_string)

  if (number > last_batch_number):
    last_batch_number = number

# Load and prepare test data
print(f"{Fore.CYAN}Loading data/batches/{last_batch_number}.csv")
test_data = pd.read_csv(f"data/batches/{last_batch_number}.csv")
test_data = prepare(test_data)

X = test_data.iloc[:,:-1]
y = test_data.iloc[:,-1]

# Evaluate
predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)
print(f'{Fore.YELLOW}Accuracy on new data: {Style.RESET_ALL}{accuracy * 100}%')

has_drift = accuracy < np.mean(previous_accuracy) - 2 * np.std(previous_accuracy)

mlflow.start_run(run_name=f"Drift check ({last_batch_number}.csv)")
mlflow.log_param("accuracy", accuracy)
mlflow.log_param("drift detected", has_drift)

if not has_drift:
  print(f'{Fore.GREEN}No drift detected {Style.RESET_ALL}')
else:
  print(f'{Fore.RED}Drift detected. Running training...{Style.RESET_ALL}')

  l = [pd.read_csv(path.join(data_path, filename)) for filename in filenames]
  full_dataset = pd.concat(l, axis=0)
  full_dataset = prepare(full_dataset)

  train(full_dataset)

mlflow.end_run()
