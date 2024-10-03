import mlflow
import mlflow.sklearn
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

BASE_PATH = os.getenv('PROJECTPATH')


# This function returns the dataframe with encoded features
# @param dataframe - initial DataFrame
# @param features_names - names of features (columns) that need to be encoded
# @param encoder - encoder that should be used
# @return new_dataframe - new DataFrame with encoded features

def encode_features_one_hot(dataframe, features_names, encoder):
    new_features = encoder.transform(dataframe[features_names])
    new_columns_df = pd.DataFrame(new_features, columns=encoder.get_feature_names_out(features_names))
    dataframe.reset_index(drop=True, inplace=True)

    new_columns_df.reset_index(drop=True, inplace=True)
    new_dataframe = pd.concat([dataframe, new_columns_df], axis=1)
    new_dataframe.drop(features_names, axis=1, inplace=True)
    return new_dataframe

def construct_features():
    X_train = pd.read_csv(f"{BASE_PATH}/data/processed/X_train.csv")
    X_test = pd.read_csv(f"{BASE_PATH}/data/processed/X_test.csv")
    y_train = pd.read_csv(f"{BASE_PATH}/data/processed/y_train.csv")
    y_test = pd.read_csv(f"{BASE_PATH}/data/processed/y_test.csv")

    # scaler_filename = f"{BASE_PATH}/models/scaler.save"

    # encoder_filename = f"{BASE_PATH}/models/encoder.save"
    # encoder = joblib.load(encoder_filename) 
    # scaler = joblib.load(scaler_filename)
    # Encoding features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Select columns that should be encoded (values are not numbers)
    features_names = list(X_train.select_dtypes(exclude='number').columns)
    print(features_names)

    encoder.fit(X_train[features_names])
    print('Columns with missing values in data: ', list(X_train.columns[X_train.isnull().any()]))
    X_train = encode_features_one_hot(X_train, features_names, encoder)
    X_test = encode_features_one_hot(X_test, features_names, encoder)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
    print(X_train)

    scaler_filename = f"{BASE_PATH}/models/scaler.save"
    joblib.dump(scaler, scaler_filename) 

    encoder_filename = f"{BASE_PATH}/models/encoder.save"
    joblib.dump(encoder, encoder_filename) 

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
        
        # Log hyperparameters
        # mlflow.log_param("model_type", "LinearRegression")
        # mlflow.log_param("encoder_type", "OneHotEncoder")
        # mlflow.log_param("scaler_type", "MinMaxScaler")
        params = {
            "model_type": "LinearRegression",
            "encoder_type": "OneHotEncoder",
            "scaler_type": "MinMaxScaler",
        }
        mlflow.log_params(params=params)

        model = LinearRegression()
        model.fit(X_train, y_train)

        filename = f'{BASE_PATH}/models/model.pkl'
        pickle.dump(model, open(filename, 'wb'))

        mlflow.sklearn.log_model(model, "model")
        
        # Log the model file path
        mlflow.log_artifact(filename)



def evaluate_model(X_test, y_test):
    with open(f"{BASE_PATH}/models/model.pkl", "rb") as f:
        model = pickle.load(f)
        
    # Predict the values
    y_pred = model.predict(X_test)
    error = mean_squared_error(y_test, y_pred)
    print('Mean squared error: ', error)
    r2 = r2_score(y_test, y_pred)
    print('R2 score: ', r2)
    mlflow.log_metric("mean_squared_error", error)
    mlflow.log_metric("r2_score", r2)

if __name__ == '__main__':
    experiment_name = "MLflow experiment 01"
    run_name = "run 01"
    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    print(experiment_id)
    #mlflow.set_experiment("Linear Regression Experiment")
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):  # Start MLflow run
    #with mlflow.start_run(run_name="run_1"):
        X_train, X_test, y_train, y_test = construct_features()
        train_model(X_train, y_train)
        evaluate_model(X_test, y_test)