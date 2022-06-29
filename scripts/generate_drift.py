import numpy as np
import pandas as pd
import sys
import os

if len(sys.argv) < 2:
  print("Usage: generate_drift {batch_number}")
  print("Fetches file from data/batches and puts generated data into data/batches")
  exit()

file_directory = os.path.dirname(os.path.realpath(__file__))

batch_number = sys.argv[1]
print(f"Loading input from data/batches/{batch_number}.csv")
test_data_path = os.path.join(file_directory, f"../data/batches/{batch_number}.csv")
test_data = pd.read_csv(test_data_path)

test_data.loc[:,'MinTemp'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)
test_data.loc[:,'MaxTemp'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)
test_data.loc[:,'Rainfall'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)
test_data.loc[:,'Evaporation'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)
test_data.loc[:,'Sunshine'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)
test_data.loc[:,'WindGustSpeed'] *= (1 + np.random.rand() * 0.5) if (np.random.rand() > 0.5) else (1 - np.random.rand() * 0.5)

out_data_path = os.path.join(file_directory, f"../data/batches/{batch_number}.csv")

print(f"Saving output to data/batches/{batch_number}.csv")
test_data.to_csv(out_data_path, index=False)
