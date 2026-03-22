ARG BASE_IMAGE=igortet/model-qwen-7b
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    bash \
    ca-certificates \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

WORKDIR /trainer

# Копируем все предварительно скачанные wheel-файлы
COPY packages/pip /wheels/pip
COPY requirements.txt /trainer/requirements.txt

RUN pip install -U pip setuptools wheel

# Устанавливаем только те пакеты, которые указаны в requirements.txt,
# используя локальные колёса (--find-links) и без обращения в интернет (--no-index).
# Pip сам разрешит и установит все необходимые зависимости (включая транзитивные,
# такие как urllib3, idna и т.д.) из папки /wheels/pip.
RUN pip install -r /trainer/requirements.txt --no-index --find-links /wheels/pip --break-system-packages

COPY app /trainer/app
COPY docker/entrypoint.sh /trainer/entrypoint.sh

RUN chmod +x /trainer/entrypoint.sh && \
    mkdir -p /output /cache/huggingface

ENTRYPOINT ["/trainer/entrypoint.sh"]