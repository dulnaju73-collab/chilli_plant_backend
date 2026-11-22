FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=2

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser models ./models
RUN python -c "import hashlib; from pathlib import Path; from urllib.request import urlretrieve; p=Path('/app/models/leaf_detection.tflite'); urlretrieve('https://huggingface.co/kd8811/TeaLeafNet/resolve/52b6970b4e5592b8f97c956df495f1fa64358b39/leaf_detection.tflite', p); assert hashlib.sha256(p.read_bytes()).hexdigest() == '92dafe78f2c8a4b4cf1b4447a3a4007221a42b6d0bcb521620fe41a63362d42f'; p.chmod(0o644)" \
    && chown appuser:appuser /app/models/leaf_detection.tflite
COPY --chown=appuser:appuser main.py ./main.py

USER appuser
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
