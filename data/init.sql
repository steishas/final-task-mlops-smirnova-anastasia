-- Feature Store
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    batch_id VARCHAR(255),
    "MedInc" FLOAT,
    "HouseAge" FLOAT,
    "AveRooms" FLOAT,
    "AveBedrms" FLOAT,
    "Population" FLOAT,
    "AveOccup" FLOAT,
    "Latitude" FLOAT,
    "Longitude" FLOAT,
    "MedHouseVal" FLOAT
);

-- Журнал предсказаний
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    features_blob BYTEA,
    prediction DOUBLE PRECISION,
    model_version VARCHAR(50)
);

-- Реальные значения для расчёта MAPE
CREATE TABLE IF NOT EXISTS ground_truth (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    true_value DOUBLE PRECISION,
    features_blob BYTEA
);

-- Скейлеры
CREATE TABLE IF NOT EXISTS scalers (
    id SERIAL PRIMARY KEY,
    scaler_name VARCHAR(50) NOT NULL,
    scaler_bytes BYTEA NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);