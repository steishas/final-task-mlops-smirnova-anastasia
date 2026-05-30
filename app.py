"""
ML-сервис прогнозирования стоимости жилья с мониторингом через Prometheus
"""
import time
import logging
from contextlib import asynccontextmanager
import os

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Gauge, Histogram, Info, make_asgi_app
from pydantic import BaseModel

import pickle

from fastapi import File, UploadFile
import io

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "CaliforniaHousingRegressor"
MODEL_STAGE = "Production"

FEATURES = [
    "MedInc", "HouseAge", "AveRooms",
    "AveBedrms", "Population", "AveOccup",
    "Latitude", "Longitude",
]

# Метрики Prometheus
LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

MAPE_GAUGE = Gauge(
    "model_mape",
    "Current rolling MAPE of the model on production data",
)

MODEL_INFO = Info("model_info", "Current model information")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Попытка загрузить из Mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        app.state.model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        logger.info("Модель загружена из MLflow")
    except Exception:
        logger.warning("Не удалось загрузить из MLflow, пробую локальный файл")
        if os.getenv("USE_LOCAL_MODEL", "false").lower() == "true":
            with open("model.pkl", "rb") as f:
                app.state.model = pickle.load(f)
            logger.info("Модель загружена из локального файла")
        else:
            app.state.model = None

    # Информация о модели
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if latest_versions:
            latest_version = latest_versions[0]
            MODEL_INFO.info({
                "name": MODEL_NAME,
                "stage": MODEL_STAGE,
                "version": latest_version.version,
                "run_id": latest_version.run_id,
            })
    except Exception as e:
        logger.warning(f"Не удалось получить информацию о модели: {e}")

    yield


app = FastAPI(
    title="California Housing Price Predictor",
    description="ML-сервис прогнозирования стоимости жилья",
    version="2.0.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


# Схемы запросов
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.556,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }
    }


class MapeUpdate(BaseModel):
    mape: float


@app.get("/health")
async def health():
    """Проверка здоровья сервиса"""
    model_loaded = app.state.model is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_name": MODEL_NAME,
    }


@app.get("/test/slow")
async def test_slow(delay: float = 0.5):
    """
    Тестовый эндпоинт для имитации высокой latency.
    По умолчанию добавляет задержку 0.5 секунды.
    """
    if delay > 0:
        time.sleep(delay)
    LATENCY.labels(endpoint="test_slow").observe(delay)
    return {"status": "ok", "message": f"Responded after {delay} seconds"}
    
@app.post("/predict")
async def predict(features: HouseFeatures):
    """Предсказание для одного объекта"""
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    start = time.time()
    try:
        df = pd.DataFrame([features.model_dump()])
        prediction = float(app.state.model.predict(df[FEATURES])[0])

        LATENCY.labels(endpoint="predict").observe(time.time() - start)

        return {
            "prediction": prediction,
            "model": MODEL_NAME,
            "latency_ms": round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Принимает CSV-файл с признаками, возвращает предсказания."""
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    start = time.time()
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Отсутствуют признаки: {missing}")

        predictions = app.state.model.predict(df[FEATURES])
        
        LATENCY.labels(endpoint="predict_batch").observe(time.time() - start)

        return JSONResponse(content={
            "predictions": predictions.tolist(),
            "n_records": len(df),
            "latency_ms": round((time.time() - start) * 1000, 2)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_mape")
async def update_mape(payload: MapeUpdate):
    """Обновляет значение MAPE в Prometheus (вызывается из Airflow)"""
    MAPE_GAUGE.set(payload.mape)
    logger.info(f"MAPE обновлён: {payload.mape:.2f}%")
    return {"status": "ok", "mape": payload.mape}


@app.post("/admin/reload")
async def admin_reload():
    """Перезагружает модель из MLflow без перезапуска сервиса (вызывается из Airflow)"""
    try:
        new_model = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}/Production"
        )
        app.state.model = new_model
        logger.info("Модель успешно перезагружена из MLflow")

        # Обновляем информацию о модели
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            latest_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
            MODEL_INFO.info({
                "name": MODEL_NAME,
                "stage": "Production",
                "version": latest_version.version,
                "run_id": latest_version.run_id,
            })
        except Exception as e:
            logger.warning(f"Не удалось обновить информацию о модели: {e}")

        return {"status": "ok", "message": "Модель перезагружена"}
    except Exception as e:
        logger.error(f"Ошибка перезагрузки модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))