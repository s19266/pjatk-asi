from colorama import Fore, Style
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
import mlflow
import mlflow.sklearn
import numpy as np

def train(dataset):
  x_train, x_test, y_train, y_test = train_test_split(
    dataset.iloc[:,:-1],
    dataset.iloc[:,-1],
    test_size=0.2,
    random_state=42
  )

  def log_output(model_name, model, accuracy, best_params):
    print()
    print(Fore.BLUE + model_name)

    print(Fore.MAGENTA + f"  Parameters:")
    for (name, value) in best_params.items():
      mlflow.log_param(name, value)
      print(Fore.MAGENTA + f"    {name}: {Style.RESET_ALL}{value}")

    print(Fore.YELLOW + f"  Accuracy: {Style.RESET_ALL}{accuracy * 100}%")
    print(Style.RESET_ALL)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

  best_accuracy = 0
  best_model = None
  best_model_name = ""
  best_model_params = ""

  # LogisticRegression
  with mlflow.start_run(run_name="Logistic Regression", nested=True):
    param_grid = [
      {
        'C' : np.logspace(-40, 40, 50),
      }
    ]

    clf = GridSearchCV(LogisticRegression(max_iter=2000, solver='liblinear'), param_grid=param_grid, cv=3)
    clf.fit(x_train, y_train)
    model = clf.best_estimator_

    predicted = model.predict(x_test)

    accuracy = accuracy_score(y_test, predicted)

    best_accuracy = accuracy
    best_model = model
    best_model_name = "LogisticRegression"
    best_model_params = clf.best_params_

    log_output("LogisticRegression", model=model, accuracy=accuracy, best_params=clf.best_params_)

  # Gaussian Naive Bayes
  with mlflow.start_run(run_name="Gaussian Naive Bayes", nested=True):
    param_grid = {
      'var_smoothing': np.logspace(0, -9, num=100)
    }

    clf = GridSearchCV(GaussianNB(), param_grid=param_grid, cv=3)
    clf.fit(x_train, y_train)
    model = clf.best_estimator_

    predicted = model.predict(x_test)

    accuracy = accuracy_score(y_test, predicted)

    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_model = model
      best_model_name = "GaussianNB"
      best_model_params = clf.best_params_

    log_output("GaussianNB", model=model, accuracy=accuracy, best_params=clf.best_params_)

  # Random Forest
  with mlflow.start_run(run_name="RandomForest", nested=True):
    param_grid = {
      'n_estimators': np.logspace(0, 3, num=10, dtype = int),
      'max_features': ['sqrt', 'log2']
    }

    clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=3)
    clf.fit(x_train, y_train)
    model = clf.best_estimator_

    predicted = model.predict(x_test)

    accuracy = accuracy_score(y_test, predicted)

    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_model = model
      best_model_name = "RandomForestClassifier"
      best_model_params = clf.best_params_

    log_output("RandomForestClassifier", model=model, accuracy=accuracy, best_params=clf.best_params_)

  with mlflow.start_run(run_name="Best", nested=True):
    mlflow.log_param("name", best_model_name)
    log_output("Best", model=best_model, accuracy=best_accuracy, best_params=best_model_params)
