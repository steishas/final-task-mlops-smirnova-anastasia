import os
import tempfile
import logging
import numpy as np
import pandas as pd
import requests
from mlflow.tracking import MlflowClient
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
APP_URL = os.getenv("APP_URL", "http://app:8000")
EXPERIMENT_NAME = "california_housing_pipeline"
BATCH_SIZE = 100
NOISE_LEVEL = 0.02
TARGET_NOISE_LEVEL = 0.05
MAPE_THRESHOLD = 20.0
SEED = 42

def get_latest_dataset():
    """Загружает последний полный датасет из MLflow или создаёт из sklearn."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    artifact_path = "raw_data/california_housing.csv"
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=10
    )
    for run in runs:
        artifacts = client.list_artifacts(run.info.run_id)
        if any(artifact_path in a.path for a in artifacts):
            logger.info(f"Загружен датасет из run {run.info.run_id}")
            local_dir = tempfile.mkdtemp()
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=artifact_path,
                dst_path=local_dir
            )
            return pd.read_csv(local_path)

    # fallback
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    
    tmp_path = os.path.join(tempfile.gettempdir(), "california_housing.csv")
    df.to_csv(tmp_path, index=False)
    with mlflow.start_run(experiment_id=experiment_id, run_name="init_dataset") as run:
        mlflow.log_artifact(tmp_path, artifact_path="raw_data")
    os.unlink(tmp_path)
    return df

def add_noise(df, noise_level=NOISE_LEVEL, target_col=None, target_noise_level=TARGET_NOISE_LEVEL, rng=None):
    """Добавляет шум к признакам и опционально к таргету."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    df_noisy = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue
        std = df[col].std()
        noise = rng.normal(0, noise_level * std, size=len(df))
        df_noisy[col] = df[col] + noise
    if target_col and target_col in df.columns:
        std_targ = df[target_col].std()
        noise_targ = rng.normal(0, target_noise_level * std_targ, size=len(df))
        df_noisy[target_col] = df[target_col] + noise_targ
    return df_noisy

def compute_mape(y_true, y_pred):
    """Mean Absolute Percentage Error (в %)."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def monitor_and_retrain():
    logger.info("Запуск мониторинга MAPE...")
   
    df_full = get_latest_dataset()
    
    # Тестовый батч с шумом
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(df_full), size=BATCH_SIZE, replace=False)
    batch = df_full.iloc[indices].reset_index(drop=True)
    target_col = "MedHouseVal"

    batch_noisy = add_noise(batch, noise_level=NOISE_LEVEL, target_col=target_col,
                            target_noise_level=TARGET_NOISE_LEVEL, rng=rng)
    
    # Инференс
    features = batch_noisy.drop(columns=[target_col])
    
    csv_buffer = features.to_csv(index=False)
    try:
        resp = requests.post(f"{APP_URL}/predict_batch", files={"file": ("batch.csv", csv_buffer)})
        if resp.status_code != 200:
            logger.error(f"Ошибка инференса: {resp.status_code} {resp.text}")
            return
        predictions = resp.json()["predictions"]
    except Exception as e:
        logger.error(f"Не удалось выполнить инференс: {e}")
        return

    # Вычисляем MAPE
    y_true = batch_noisy[target_col].values
    mape = compute_mape(y_true, predictions)
    logger.info(f"Текущий MAPE на батче: {mape:.2f}%")

    # Отправляем MAPE в Prometheus через FastAPI
    try:
        requests.post(f"{APP_URL}/update_mape", json={"mape": mape})
    except Exception as e:
        logger.warning(f"Не удалось обновить MAPE: {e}")

    # Проверка порога
    if mape > MAPE_THRESHOLD:
        logger.info(f"MAPE превысил порог {MAPE_THRESHOLD}%, требуется проверка")

        df_new_full = add_noise(df_full, noise_level=NOISE_LEVEL, target_col=target_col,
                                target_noise_level=TARGET_NOISE_LEVEL, rng=rng)
        # Логируем данные в MLflow
        tmp_path = os.path.join(tempfile.gettempdir(), "california_housing.csv")
        df_new_full.to_csv(tmp_path, index=False)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
        with mlflow.start_run(experiment_id=experiment_id, run_name="new_data_for_retrain") as run:
            mlflow.log_artifact(tmp_path, artifact_path="raw_data")
            logger.info(f"Новый датасет залогирован в run {run.info.run_id}")
        os.unlink(tmp_path)

    else:
        logger.info("MAPE в норме, переобучение не требуется.")

if __name__ == "__main__":
    monitor_and_retrain()