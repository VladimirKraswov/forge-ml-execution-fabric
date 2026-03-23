#!/bin/bash
set -e

# Убедимся, что папка для вывода существует
mkdir -p ./output

# Запуск контейнера
docker run -it --rm \
  --gpus all \
  --shm-size 16g \
  -v $(pwd)/data:/data \
  -v $(pwd)/job.local.json:/configs/job.local.json \
  -v $(pwd)/output:/output \
  -e CONFIG_SOURCE=local \
  -e CONFIG_REF=/configs/job.local.json \
  igortet/itk-executor-trainer:latest