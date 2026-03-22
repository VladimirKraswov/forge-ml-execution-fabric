## 1. Callback: status

Используется для крупных смен стадий:

* started
* running
* finished
* failed

### Endpoint

`POST /api/jobs/status`

### Тело запроса

```json
{
  "job_id": "job_qwen25_05b_demo_remote",
  "job_name": "qwen25-05b-demo-remote",
  "event": "status",
  "timestamp": 1773926400.123,
  "status": "running",
  "stage": "prepare_assets",
  "progress": 5,
  "message": "Preparing datasets and remote assets",
  "logs": null,
  "extra": {}
}
```

### Поля

```json
{
  "job_id": "string",
  "job_name": "string",
  "event": "status",
  "timestamp": "number (unix seconds)",
  "status": "started | running | finished | failed",
  "stage": "string | null",
  "progress": "number 0..100 | null",
  "message": "string | null",
  "logs": "string | null",
  "extra": "object"
}
```

### Что backend должен делать

* находить job по `job_id`
* обновлять:

  * общий статус
  * текущую стадию
  * progress
  * last_message
  * updated_at
* если `status=failed`, сохранить `logs` как diagnostic tail

### Ответ

```json
{
  "ok": true
}
```

---

## 2. Callback: progress

Используется часто, для живого прогресса во время train/eval.

### Endpoint

`POST /api/jobs/progress`

### Тело запроса

```json
{
  "job_id": "job_qwen25_05b_demo_remote",
  "job_name": "qwen25-05b-demo-remote",
  "event": "progress",
  "timestamp": 1773926412.456,
  "status": "running",
  "stage": "training",
  "progress": 37.5,
  "message": "trainer log",
  "extra": {
    "step": 120,
    "epoch": 0.42,
    "loss": 1.1834,
    "grad_norm": 0.91,
    "learning_rate": 0.000082
  }
}
```

### Поля

```json
{
  "job_id": "string",
  "job_name": "string",
  "event": "progress",
  "timestamp": "number",
  "status": "running",
  "stage": "string",
  "progress": "number 0..100 | null",
  "message": "string | null",
  "extra": "object"
}
```

### Что backend должен делать

* обновлять live progress
* писать историю progress events в отдельную таблицу или log stream
* по желанию агрегировать последние train metrics в UI

---

## 3. Callback: final

Финальный результат job целиком.

### Endpoint

`POST /api/jobs/final`

### Тело запроса при успехе

```json
{
  "job_id": "job_qwen25_05b_demo_remote",
  "job_name": "qwen25-05b-demo-remote",
  "event": "final",
  "timestamp": 1773926999.111,
  "status": "finished",
  "result": {
    "status": "success",
    "job_id": "job_qwen25_05b_demo_remote",
    "job_name": "qwen25-05b-demo-remote",
    "started_at": "2026-03-19T13:00:00+00:00",
    "finished_at": "2026-03-19T13:09:59+00:00",
    "config_source": "https://example.com/job.json",
    "training": {
      "status": "success",
      "job_name": "qwen25-05b-demo-remote",
      "base_model": "/models/qwen25-05b",
      "lora_dir": "/output/qwen25-05b-demo-remote/lora/qwen25-05b-demo-remote",
      "merged_dir": "/output/qwen25-05b-demo-remote/merged/qwen25-05b-demo-remote",
      "checkpoint_dir": "/output/qwen25-05b-demo-remote/checkpoints/qwen25-05b-demo-remote",
      "metrics_path": "/output/qwen25-05b-demo-remote/metrics/qwen25-05b-demo-remote.train_metrics.json",
      "history_path": "/output/qwen25-05b-demo-remote/metrics/qwen25-05b-demo-remote.train_history.json",
      "train_summary_path": "/output/qwen25-05b-demo-remote/metrics/qwen25-05b-demo-remote.train_summary.json",
      "summary": {
        "job_name": "qwen25-05b-demo-remote",
        "train_rows": 10000,
        "validation_rows": 500,
        "method": "qlora",
        "base_model": "/models/qwen25-05b",
        "load_in_4bit": true,
        "bf16": true,
        "merged_saved": true,
        "train_runtime": 522.4,
        "train_samples_per_second": 19.1,
        "train_steps_per_second": 0.52,
        "train_loss": 0.947,
        "final_loss": 0.913
      }
    },
    "evaluation": {
      "enabled": true,
      "target": "auto",
      "model": "/output/qwen25-05b-demo-remote/merged/qwen25-05b-demo-remote",
      "summary": {
        "model": "/output/qwen25-05b-demo-remote/merged/qwen25-05b-demo-remote",
        "samples": 100,
        "parseSuccessRate": 0.96,
        "mae": 0.41,
        "rmse": 0.67,
        "exactRate": 0.58,
        "within1Rate": 0.95,
        "within2Rate": 1.0,
        "meanSignedError": -0.08,
        "avgPredictedScore": 3.82,
        "parseErrors": 4,
        "inferenceErrors": 0,
        "emptyResponses": 1
      },
      "summary_json_path": "/output/qwen25-05b-demo-remote/evaluation/qwen25-05b-demo-remote/summary.json",
      "result_json_path": "/output/qwen25-05b-demo-remote/evaluation/qwen25-05b-demo-remote/result.json",
      "summary_csv_path": "/output/qwen25-05b-demo-remote/evaluation/qwen25-05b-demo-remote/summary.csv",
      "detailed_csv_path": "/output/qwen25-05b-demo-remote/evaluation/qwen25-05b-demo-remote/detailed.csv"
    },
    "artifacts": {
      "log_file": "/output/qwen25-05b-demo-remote/logs/trainer.log",
      "effective_config_path": "/output/qwen25-05b-demo-remote/logs/effective-job.json",
      "result_path": "/output/qwen25-05b-demo-remote/job-result.json"
    },
    "uploads": {
      "logs": {
        "url": "https://example.com/api/jobs/upload/logs",
        "path": "/output/qwen25-05b-demo-remote/logs/trainer.log"
      },
      "lora_archive": {
        "url": "https://example.com/api/jobs/upload/lora",
        "archive_path": "/output/qwen25-05b-demo-remote/qwen25-05b-demo-remote.lora.tar.gz"
      }
    }
  }
}
```

### Тело запроса при ошибке

```json
{
  "job_id": "job_qwen25_05b_demo_remote",
  "job_name": "qwen25-05b-demo-remote",
  "event": "final",
  "timestamp": 1773926999.111,
  "status": "failed",
  "result": {
    "status": "failed",
    "job_id": "job_qwen25_05b_demo_remote",
    "job_name": "qwen25-05b-demo-remote",
    "started_at": "2026-03-19T13:00:00+00:00",
    "finished_at": "2026-03-19T13:02:11+00:00",
    "config_source": "https://example.com/job.json",
    "error": "Evaluation dataset is empty after normalization",
    "artifacts": {
      "log_file": "/output/qwen25-05b-demo-remote/logs/trainer.log",
      "effective_config_path": "/output/qwen25-05b-demo-remote/logs/effective-job.json",
      "result_path": "/output/qwen25-05b-demo-remote/job-result.json"
    }
  }
}
```

### Что backend должен делать

* помечать job завершённым
* сохранять итоговый payload целиком
* разбирать `training.summary`
* разбирать `evaluation.summary`
* связывать загруженные артефакты с job

---

## 4. Upload endpoints

Runner отправляет файлы multipart/form-data.

### Общий формат upload request

`POST /api/jobs/upload/<artifact-type>`

`Content-Type: multipart/form-data`

Поля формы:

* `job_id`
* `job_name`
* `artifact_type`
* `file`

### Пример

Поле `file` содержит бинарник файла или tar.gz.

### Универсальный ответ

```json
{
  "ok": true,
  "artifact_id": "art_123",
  "job_id": "job_qwen25_05b_demo_remote",
  "artifact_type": "lora_archive",
  "storage_key": "jobs/job_qwen25_05b_demo_remote/lora.tar.gz",
  "download_url": "https://cdn.example.com/jobs/job_qwen25_05b_demo_remote/lora.tar.gz"
}
```

---

## 5. Какие upload endpoints нужны

### Логи

`POST /api/jobs/upload/logs`

Принимает:

* `trainer.log`

Сохраняет как:

* plain text log artifact

### Effective config

`POST /api/jobs/upload/config`

Принимает:

* `effective-job.json`

Сохраняет как:

* исходный эффективный конфиг job

### Итоговый summary

`POST /api/jobs/upload/summary`

Принимает:

* `job-result.json`

Сохраняет как:

* финальный канонический результат job

### Train metrics

`POST /api/jobs/upload/train-metrics`

Принимает:

* `*.train_metrics.json`

### Train history

`POST /api/jobs/upload/train-history`

Принимает:

* `*.train_history.json`

### Eval summary

`POST /api/jobs/upload/eval-summary`

Принимает:

* `summary.json` с метриками eval

### Eval details

`POST /api/jobs/upload/eval-details`

Принимает:

* `detailed.csv`

### LoRA archive

`POST /api/jobs/upload/lora`

Принимает:

* tar.gz с LoRA adapter

### Merged archive

`POST /api/jobs/upload/merged`

Принимает:

* tar.gz с merged model

### Full archive

`POST /api/jobs/upload/full-archive`

Принимает:

* tar.gz со всем output job

---

## 6. Что лучше хранить в БД

Минимальная таблица `jobs`:

```json
{
  "id": "job_qwen25_05b_demo_remote",
  "name": "qwen25-05b-demo-remote",
  "status": "running",
  "stage": "training",
  "progress": 37.5,
  "message": "trainer log",
  "created_at": "timestamp",
  "updated_at": "timestamp",
  "started_at": "timestamp|null",
  "finished_at": "timestamp|null",
  "error": "string|null",
  "result_json": "json|null"
}
```

Таблица `job_events`:

```json
{
  "id": "evt_xxx",
  "job_id": "job_qwen25_05b_demo_remote",
  "event": "status|progress|final",
  "payload": "json",
  "created_at": "timestamp"
}
```

Таблица `job_artifacts`:

```json
{
  "id": "art_xxx",
  "job_id": "job_qwen25_05b_demo_remote",
  "artifact_type": "logs|config|summary|train_metrics|train_history|eval_summary|eval_details|lora_archive|merged_archive|full_archive",
  "filename": "string",
  "storage_key": "string",
  "download_url": "string|null",
  "content_type": "string|null",
  "size_bytes": "number|null",
  "created_at": "timestamp"
}
```

---

## 7. Рекомендуемая логика статусов

Я бы использовал такой lifecycle:

* `queued`
* `started`
* `running`
* `finished`
* `failed`

И такие stage:

* `bootstrap`
* `hf_login`
* `prepare_assets`
* `load_model`
* `load_dataset`
* `training`
* `save_lora`
* `merge_lora`
* `train_completed`
* `evaluation_prepare`
* `evaluation`
* `evaluation_completed`
* `finished`
* `failed`

---

## 8. Авторизация

Лучше передавать bearer token в конфиге:

```json
{
  "reporting": {
    "status": {
      "enabled": true,
      "url": "https://example.com/api/jobs/status",
      "auth": {
        "bearer_token": "secret-token"
      }
    }
  },
  "upload": {
    "enabled": true,
    "target": "url",
    "auth": {
      "bearer_token": "secret-token"
    }
  }
}
```

Backend должен принимать:

```http
Authorization: Bearer secret-token
```

---

## 9. Идемпотентность

Полезно, чтобы backend спокойно переживал ретраи.

Для этого:

* `job_id` должен быть уникальным и приходить из твоего сервиса
* повторный `status/progress/final` по тому же `job_id` не должен ломать запись
* upload можно делать либо:

  * overwrite по `(job_id, artifact_type)`
  * либо versioned storage, но в БД держать “последнюю актуальную версию”

---

## 10. Что я советую сделать на твоём backend сразу

1. Один endpoint на status
2. Один endpoint на progress
3. Один endpoint на final
4. Один универсальный upload endpoint вида
   `POST /api/jobs/upload/:artifactType`
5. В UI показывать:

   * status
   * stage
   * progress
   * train summary
   * eval summary
   * список артефактов

---

## 11. Минимальный backend response contract

Для всех callback endpoints:

```json
{
  "ok": true
}
```

Для всех upload endpoints:

```json
{
  "ok": true,
  "artifact_id": "art_xxx",
  "download_url": "https://cdn.example.com/path/to/file"
}
```

Коды:

* `200` или `201` — успех
* `400` — невалидный payload
* `401/403` — auth error
* `404` — job not found, если ты это хочешь строго контролировать
* `500` — временная ошибка, runner попробует/сможет быть перезапущен на уровне orchestration

---

## 12. Пример одной job-конфигурации с backend-контрактом

```json
{
  "job_id": "job_20260319_001",
  "job_name": "qwen25-math-v3",
  "reporting": {
    "status": {
      "enabled": true,
      "url": "https://api.example.com/jobs/status",
      "auth": {
        "bearer_token": "secret-token"
      }
    },
    "progress": {
      "enabled": true,
      "url": "https://api.example.com/jobs/progress",
      "auth": {
        "bearer_token": "secret-token"
      }
    },
    "final": {
      "enabled": true,
      "url": "https://api.example.com/jobs/final",
      "auth": {
        "bearer_token": "secret-token"
      }
    }
  },
  "upload": {
    "enabled": true,
    "target": "url",
    "auth": {
      "bearer_token": "secret-token"
    },
    "url_targets": {
      "logs_url": "https://api.example.com/jobs/upload/logs",
      "summary_url": "https://api.example.com/jobs/upload/summary",
      "lora_archive_url": "https://api.example.com/jobs/upload/lora",
      "merged_archive_url": "https://api.example.com/jobs/upload/merged",
      "eval_summary_url": "https://api.example.com/jobs/upload/eval-summary",
      "eval_details_url": "https://api.example.com/jobs/upload/eval-details"
    }
  }
}