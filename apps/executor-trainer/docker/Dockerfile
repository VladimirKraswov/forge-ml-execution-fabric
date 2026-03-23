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

WORKDIR /trainer

COPY requirements.txt /trainer/requirements.txt
COPY packages /trainer/packages
COPY docker/install_with_local_wheels.sh /trainer/install_with_local_wheels.sh

RUN chmod +x /trainer/install_with_local_wheels.sh && \
    mkdir -p /trainer/packages/pip && \
    echo "==> copied local wheels:" && \
    find /trainer/packages/pip -maxdepth 1 -type f -name '*.whl' | sort | head -200

# train env
RUN python3 -m venv /opt/venv
RUN /opt/venv/bin/pip install -U pip setuptools wheel
RUN /trainer/install_with_local_wheels.sh /opt/venv/bin/python --requirements /trainer/requirements.txt /trainer/packages/pip

# separate vLLM env
RUN python3 -m venv /opt/vllm-venv
RUN /opt/vllm-venv/bin/pip install -U pip setuptools wheel
RUN /trainer/install_with_local_wheels.sh /opt/vllm-venv/bin/python --package vllm /trainer/packages/pip

COPY app /trainer/app
COPY docker/entrypoint.sh /trainer/entrypoint.sh

RUN chmod +x /trainer/entrypoint.sh && \
    mkdir -p /output /cache/huggingface

ENTRYPOINT ["/trainer/entrypoint.sh"]