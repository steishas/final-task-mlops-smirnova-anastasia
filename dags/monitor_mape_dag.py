import sys
sys.path.insert(0, '/opt/airflow')
import monitor_and_retrain

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='monitor_mape',
    default_args=default_args,
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['monitoring'],
    description='Мониторинг MAPE и подготовка данных для переобучения',
) as dag:

    monitor_task = PythonOperator(
        task_id='monitor_mape_task',
        python_callable=monitor_and_retrain.monitor_and_retrain,
    )