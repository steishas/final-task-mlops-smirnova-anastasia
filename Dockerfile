FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries=5 -r requirements.txt

# RUN pip install --no-cache-dir --default-timeout=120 --retries=5 \
#     -i https://mirrors.aliyun.com/pypi/simple/ \
#     --trusted-host mirrors.aliyun.com \
#     -r requirements.txt

COPY app.py .
COPY model.pkl .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]