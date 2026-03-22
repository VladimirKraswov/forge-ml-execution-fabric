# trainer-service

Одноразовый job-runner для fine-tuning на Vast.ai.

Сервис не хранит состояние как backend-приложение. Он запускается на один job, читает один JSON-конфиг, выполняет pipeline и выгружает наружу:
- статусы;
- прогресс;
- логи;
- LoRA;
- merged model;
- train metrics;
- evaluation results.

## Pipeline

1. Загружает JSON-конфиг из локального файла или по URL.
2. Логинится в Hugging Face по `HF_TOKEN`, если токен передан.
3. При необходимости скачивает train/val/eval датасеты по URL.
4. Запускает fine-tuning через Unsloth.
5. Сохраняет LoRA adapter.
6. Опционально сохраняет merged 16-bit модель.
7. Опционально запускает evaluation после обучения.
8. Опционально:
   - пушит LoRA / merged model в Hugging Face;
   - загружает артефакты по URL;
   - отправляет callbacks со статусами и прогрессом.
9. Сохраняет итог в `job-result.json`.

## Ключевые требования, которые покрыты

- Конфиг может лежать по URL.
- Модель может быть уже встроена в Docker image (`model.source=local`).
- Есть внешние callbacks для:
  - started/running/finished/failed;
  - progress;
  - final result.
- Есть отдельная выгрузка:
  - logs;
  - effective config;
  - train metrics / history;
  - eval summary / details;
  - LoRA archive;
  - merged archive;
  - full archive.
- Есть post-train evaluation с метриками:
  - `model`
  - `samples`
  - `parseSuccessRate`
  - `mae`
  - `rmse`
  - `exactRate`
  - `within1Rate`
  - `within2Rate`
  - `meanSignedError`
  - `avgPredictedScore`
  - `parseErrors`
  - `inferenceErrors`
  - `emptyResponses`

## Запуск

### Через env

```bash
docker run --rm --gpus all \
  -e HF_TOKEN=hf_xxx \
  -e CONFIG_SOURCE=remote \
  -e CONFIG_REF=https://example.com/job.json \
  -v $(pwd)/output:/output \
  trainer-service