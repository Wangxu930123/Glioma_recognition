ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COMPETITION_WORKSPACE=/2026aicompetition/workspace

WORKDIR /opt/competition
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x start.sh

EXPOSE 8000
CMD ["./start.sh"]

