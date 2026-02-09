FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_md

COPY . .
RUN adduser --disabled-password --gecos '' appuser
USER appuser

CMD ["celery", "-A", "app.celery_app", "worker", "--beat", "--loglevel=info", "--concurrency=1", "--schedule=/tmp/celerybeat-schedule"]
