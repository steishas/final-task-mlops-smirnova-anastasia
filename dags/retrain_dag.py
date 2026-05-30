import sys
sys.path.append('/opt/airflow')
import train_model

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='retrain_model_manual',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['manual', 'retrain'],
    description='Ручное переобучение модели после алерта',
) as dag:

    def retrain():
        train_model.main()

    retrain_task = PythonOperator(
        task_id='retrain_task',
        python_callable=retrain,
    )