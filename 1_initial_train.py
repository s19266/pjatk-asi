import warnings
import mlflow
from utils.train import train
from utils.prepare import prepare
import logging
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

csv_url = ("./data/batches/0.csv")
dataset = pd.read_csv(csv_url, sep=',')

prepared_dataset = prepare(dataset)

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.start_run(run_name="Initial Training (0.csv)")
train(prepared_dataset)
mlflow.end_run()
