FROM python:3.10

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive \
  && apt-get install -y net-tools netcat-traditional curl ffmpeg libsm6 libxext6 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir --parents /opt/app
COPY pyproject.toml /opt/app/pyproject.toml
COPY poetry.lock /opt/app/poetry.lock
COPY poetry.toml /opt/app/poetry.toml

WORKDIR /opt/app

RUN pip install poetry \
    && poetry install --only main --no-root

COPY scripts /opt/app/scripts
COPY src /opt/app/src
COPY train.py /opt/app/train.py
COPY yolov8n.pt /opt/app/yolov8n.pt

COPY entrypoint.sh /opt/app/entrypoint.sh

RUN chmod +x /opt/app/entrypoint.sh
