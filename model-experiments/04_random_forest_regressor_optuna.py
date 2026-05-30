import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("california_housing_random_forest")

import optuna
import pandas as pd
from sklearn import metrics, model_selection
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def evaluate_regression(y_true, y_pred):
    """
    Оценивает качество регрессионной модели.

    Параметры:
        y_true (np.array): Фактические значения
        y_pred (np.array): Предсказанные значения

    Возвращает:
        pd.DataFrame: Датафрейм с метриками
    """
    # Расчет метрик
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = metrics.r2_score(y_true, y_pred)

    # Создание датафрейма с результатами
    evaluation_df = pd.DataFrame(
        {
            "MSE": [f"{mse:.2f}"],
            "MAE": [f"{mae:.2f}"],
            "MAPE": [f"{mape:.2f}"],
            "R^2": [f"{r2:.4f}"],
        }
    )

    return evaluation_df, mse, mae, mape, r2


data = fetch_california_housing(as_frame=True)
data = data.frame.copy()

# Разделяем на X и y
X = data.copy().drop(columns="MedHouseVal")
y = data["MedHouseVal"].copy()

# Разделяем на train/test
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=SEED
)


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
    }

    with mlflow.start_run(run_name=f"trial_{trial.number}"):

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "tree",
                    RandomForestRegressor(
                        **params, random_state=SEED, n_jobs=-1
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        # Предсказание
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Метрики
        evaluation_train_df, mse_train, mae_train, mape_train, r2_train = (
            evaluate_regression(y_train, y_pred_train)
        )
        evaluation_test_df, mse_test, mae_test, mape_test, r2_test = (
            evaluate_regression(y_test, y_pred_test)
        )

        # Логирование параметров
        mlflow.log_params(params)

        # Логирование метрик
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

        # Логирование пайплайна
        mlflow.sklearn.log_model(pipeline, "model")

        return mape_test


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

best_params = study.best_params

with mlflow.start_run(run_name=f"RandomForestRegressor_BestTrial"):

    best_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "tree",
                RandomForestRegressor(
                    **best_params, random_state=SEED, n_jobs=-1
                ),
            ),
        ]
    )

    best_pipeline.fit(X_train, y_train)

    # Предсказание
    y_pred_train = best_pipeline.predict(X_train)
    y_pred_test = best_pipeline.predict(X_test)

    # Метрики
    evaluation_train_df, mse_train, mae_train, mape_train, r2_train = (
        evaluate_regression(y_train, y_pred_train)
    )
    evaluation_test_df, mse_test, mae_test, mape_test, r2_test = (
        evaluate_regression(y_test, y_pred_test)
    )

    # Логирование параметров
    mlflow.log_params(best_params)
    mlflow.log_param("random_state", SEED)
    mlflow.log_param("n_jobs", -1)

    # Логирование метрик
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

    # Логирование пайплайна
    mlflow.sklearn.log_model(best_pipeline, "model")

    print("MSE, MAE, MAPE, R^2")
    print("Для train:")
    print(evaluation_train_df)

    print()
    print("Для test:")
    print(evaluation_test_df)
