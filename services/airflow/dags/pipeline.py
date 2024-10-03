import os
from datetime import datetime
from airflow import DAG
from airflow.decorators import task
import sys

# Import functions from your existing scripts
from prepare_data import read_data, clean_data, split_data
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.python import PythonOperator


BASE_PATH = os.getenv('PROJECTPATH')
sys.path.insert(0, BASE_PATH+"/code/models")
from train_model import main
with DAG(
   dag_id="pipeline", 
		 start_date=datetime(2024, 10, 2),
         catchup=False,
		 schedule="*/5 * * * *"
) as parent_dag:

    trigger_child_dag = TriggerDagRunOperator(
        task_id='trigger_child_dag',
        trigger_dag_id='data_preparation', 
        conf={"key": "value"},       
        wait_for_completion=False     
    )

    run_train = PythonOperator(
        task_id='train_model',
        python_callable=main,
    )

    trigger_child_dag >> run_train