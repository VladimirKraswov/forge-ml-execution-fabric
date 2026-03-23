# Локальный запуск тренировки

## Требования
- Docker Engine
- NVIDIA Container Toolkit (для GPU)
- GPU с драйверами NVIDIA (если нет GPU, уберите `--gpus all` из `run.sh`)

## Запуск
1. **Замените данные** в папке `data/` на свои файлы:
   - `train.json` – обучающая выборка (формат instruction-output)
   - `val.json` – валидационная выборка
   - `eval.jsonl` – оценочная выборка (jsonl)

2. (Опционально) **Отредактируйте** `job.local.json`, если нужно изменить параметры обучения (эпохи, batch size и т.д.).

3. Запустите скрипт:
   ```bash
   ./run.sh