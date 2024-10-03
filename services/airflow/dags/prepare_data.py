import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

from airflow import DAG
from airflow.decorators import task
import os

BASE_PATH = os.getenv('PROJECTPATH')

# A DAG represents a workflow, a collection of tasks
# This DAG is scheduled to print 'hello world' every minute starting from 01.01.2022.
with DAG(dag_id="data_preparation", 
		 start_date=datetime(2024, 10, 2),
         catchup=False,
		 schedule="*/5 * * * *") as dag:
	
    @task()
    def read_data():
        data = pd.read_csv(f"{BASE_PATH}/data/raw/car_data.csv")
        return data

    @task()
    def clean_data(**context):
        data = context['task_instance'].xcom_pull(task_ids='read_data')
        data = data.drop(['Car_Name'], axis=1)
        data.dropna(inplace=True)
        return data

    @task()
    def split_data(**context):
        data = context['task_instance'].xcom_pull(task_ids='clean_data')
        X = data.drop(['Selling_Price'], axis=1)
        y = data['Selling_Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # save the data
        X_train.to_csv(f"{BASE_PATH}/data/processed/X_train.csv", index=False)
        X_test.to_csv(f"{BASE_PATH}/data/processed/X_test.csv", index=False)
        y_train.to_csv(f"{BASE_PATH}/data/processed/y_train.csv", index=False)
        y_test.to_csv(f"{BASE_PATH}/data/processed/y_test.csv", index=False)

		
    # Set dependencies between tasks
    # First is hello task then world task
    read_data() >> clean_data() >> split_data()
