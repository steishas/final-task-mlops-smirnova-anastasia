# train_model.py
import logging
import os
import tempfile
import pickle

import mlflow
from mlflow.tracking import MlflowClient

import pandas as pd
import psycopg2
from lightgbm import LGBMRegressor
from sklearn import metrics, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# MLflow, fallback для локального запуска
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
mlflow_experiment_name = "california_housing_pipeline"

# База данных, fallback для локального запуска
host = os.getenv("DB_HOST", "localhost")
port = int(os.getenv("DB_PORT", "5001"))
database = os.getenv("DB_NAME", "feature_store")
user = os.getenv("DB_USER", "postgres_user")
password = os.getenv("DB_PASSWORD", "postgres_password")

# Гиперпараметры
SEED = 42
n_estimators = 100
learning_rate = 0.1
max_depth = -1
num_leaves = 31
n_jobs = -1

threshold_mape = 20.0
test_size = 0.2

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(y_true, y_pred):
    """
    Оценивает качество регрессионной модели.

    Параметры:
        y_true (np.array): Фактические значения
        y_pred (np.array): Предсказанные значения

    Возвращает:
        pd.DataFrame: Датафрейм с метриками
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = metrics.r2_score(y_true, y_pred)

    evaluation_df = pd.DataFrame(
        {
            "MSE": [f"{mse:.2f}"],
            "MAE": [f"{mae:.2f}"],
            "MAPE": [f"{mape:.2f}"],
            "R^2": [f"{r2:.4f}"],
        }
    )

    return evaluation_df, mse, mae, mape, r2


def load():
    """
    Загружает датасет для переобучения:
      - Ищет последний run с именем 'new_data_for_retrain' (создается при деградации MAPE).
      - Если такого нет, загружает любой последний датасет из эксперимента.
      - Если такого нет, создаёт базовый из sklearn и логирует его.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # Получаем эксперимент
    experiment = client.get_experiment_by_name(mlflow_experiment_name)
    if not experiment:
        experiment_id = client.create_experiment(mlflow_experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # Ищем run с именем new_data_for_retrain
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.runName = 'new_data_for_retrain'",
        order_by=["start_time DESC"],
        max_results=1
    )
    if runs:
        run = runs[0]
        logger.info(f"Найден датасет 'new_data_for_retrain' (run {run.info.run_id})")
        local_dir = tempfile.mkdtemp()
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id,
            artifact_path="raw_data/california_housing.csv",
            dst_path=local_dir
        )
        return pd.read_csv(local_path)

    # Ищем любой датасет в эксперименте
    artifact_path = "raw_data/california_housing.csv"
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=10
    )
    for run in runs:
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=artifact_path,
                dst_path=tempfile.mkdtemp()
            )
            logger.info(f"Загружен последний доступный датасет из run {run.info.run_id}")
            return pd.read_csv(local_path)
        except Exception:
            continue

    # Если ничего не найдено базовый датасет sklearn
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    tmp_path = os.path.join(tempfile.gettempdir(), "california_housing.csv")
    df.to_csv(tmp_path, index=False)
    with mlflow.start_run(experiment_id=experiment_id, run_name="init_dataset") as run:
        mlflow.log_artifact(tmp_path, artifact_path="raw_data")
    os.unlink(tmp_path)
    return df


def get_db_connection():
    """
    Получает объект подключения к базе данных
    """
    conn = psycopg2.connect(
        host=host, port=port, database=database, user=user, password=password
    )
    return conn


def preprocess_and_update_feature_store(df, db_connection):
    """
    Разделяет данные на train/test и сохраняет их в таблицу features
    с помощью переданного psycopg2-соединения.
    Возвращает немасштабированные X_train, X_test и y_train, y_test.
    """
    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )

    columns = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"
    ]
    train_records = [
        tuple(row[col] for col in columns)
        for _, row in pd.concat([X_train, y_train], axis=1).iterrows()
    ]
    test_records = [
        tuple(row[col] for col in columns)
        for _, row in pd.concat([X_test, y_test], axis=1).iterrows()
    ]

    insert_query = """
        INSERT INTO features
        ("MedInc", "HouseAge", "AveRooms", "AveBedrms",
         "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    with db_connection.cursor() as cur:
        cur.executemany(insert_query, train_records)
        cur.executemany(insert_query, test_records)
        db_connection.commit()

    logger.info(
        f"Сохранено в features: {len(train_records)} train + {len(test_records)} test записей"
    )

    return X_train, y_train, X_test, y_test


def train(X_train, y_train, X_test, y_test):
    """
    Создаёт Pipeline (StandardScaler + LGBMRegressor),
    обучает его на немасштабированных данных и логирует весь pipeline в MLflow.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name="LightGBM_pipeline") as run:
        run_id = run.info.run_id

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                random_state=SEED,
                n_jobs=n_jobs,
            ))
        ])

        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        _, mse_train, mae_train, mape_train, r2_train = evaluate(y_train, y_pred_train)
        _, mse_test, mae_test, mape_test, r2_test = evaluate(y_test, y_pred_test)

        mlflow.log_params(
            {
                "model_type": "LightGBM",
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "random_state": SEED,
                "n_jobs": n_jobs,
            }
        )

        mlflow.log_metrics(
            {
                "train_MSE": mse_train,
                "train_MAE": mae_train,
                "train_MAPE": mape_train,
                "train_R2": r2_train,
                "test_MSE": mse_test,
                "test_MAE": mae_test,
                "test_MAPE": mape_test,
                "test_R2": r2_test,
            }
        )

        mlflow.sklearn.log_model(pipeline, "model")

        # Сохраняем StandardScaler в Feature Store
        scaler = pipeline.named_steps['scaler']
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO scalers (scaler_name, scaler_bytes) VALUES (%s, %s)",
            ("default_scaler", pickle.dumps(scaler))
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info("StandardScaler сохранён в таблицу scalers")

    return run_id, pipeline, mape_test


def register_model(
    run_id,
    mape_test,
    threshold_mape=threshold_mape,
    model_name="CaliforniaHousingRegressor",
):
    client = MlflowClient()
    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    version = result.version
    logger.info(f"Модель зарегистрирована: {model_name} v{version}")
    client.set_model_version_tag(model_name, version, "test_MAPE", f"{mape_test:.2f}")

    if mape_test <= threshold_mape:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        for mv in all_versions:
            if mv.current_stage == "Production":
                client.transition_model_version_stage(model_name, mv.version, "Archived")
                logger.info(f"Версия {mv.version} перемещена в Archived")

        client.transition_model_version_stage(model_name, version, "Production")
        logger.info(f"Версия {version} переведена в Production")
    else:
        logger.info(f"MAPE={mape_test:.2f}% превышает порог {threshold_mape}%, продвижение отменено")

    return version


def main():
    df = load()
    db_connection = get_db_connection()
    X_tr, y_tr, X_te, y_te = preprocess_and_update_feature_store(df, db_connection)
    run_id, pipeline, mape_test = train(X_tr, y_tr, X_te, y_te)
    version = register_model(
        run_id, mape_test,
        threshold_mape=threshold_mape,
        model_name="CaliforniaHousingRegressor",
    )

    # Уведомление FastAPI о новой модели в Production
    if version:
        try:
            import requests
            app_url = os.getenv("APP_URL", "http://app:8000")
            response = requests.post(f"{app_url}/admin/reload", timeout=10)
            if response.status_code == 200:
                logger.info("FastAPI перезагрузил модель")
            else:
                logger.warning(f"FastAPI вернул {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Не удалось уведомить FastAPI: {e}")

if __name__ == "__main__":
    main()