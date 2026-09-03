FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend ./backend
RUN pip install --no-cache-dir .

RUN mkdir -p /app/data /app/logs
EXPOSE 8000
CMD ["python", "-m", "backend.run"]

