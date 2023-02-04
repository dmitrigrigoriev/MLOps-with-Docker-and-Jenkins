import pandas as pd
import numpy as np

import json
import os
from joblib import dump

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(os.environ["MLFLOW_ADDRESS"])
mlflow.set_experiment("adult-train2")

# Set path to inputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_data_file = 'train.csv'
train_data_path = os.path.join(PROCESSED_DATA_DIR, train_data_file)

# Read data
df = pd.read_csv(train_data_path, sep=",")

n_estimators = [int(x) for x in np.linspace(start=100, stop = 1000, num = 20)]
max_features = ["sqrt"]
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
min_samples_split = [2, 3, 4, 5, 6]
min_samples_leaf=[1, 2, 3, 4, 5]
bootstrap = [True, False]

# Split data into dependent and independent variables
X_train = df.drop('income', axis=1)
y_train = df['income']

random_grid = { "n_estimators": n_estimators,
                "max_features": max_features,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "bootstrap": bootstrap,
                "random_state": [42]
                 }

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter = 10, cv=3, verbose=3, random_state=42, n_jobs=1)
rf_random.fit(X_train, y_train)

df = pd.DataFrame(rf_random.cv_results_)

with mlflow.start_run():
    for index, row in df.iterrows():
        mlflow.log_param(row["params"])
        mlflow.log_metrcis("iterattion", row["iter"])
        mlflow.log_metrcis("resources", row["n_resources"])        
        mlflow.log_metrcis("score", row["mean_test_score"])
