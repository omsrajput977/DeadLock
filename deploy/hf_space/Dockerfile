# Hugging Face Spaces Docker image for DeadLock live web demo
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git git-lfs curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
RUN uv venv --python 3.13 /opt/venv

COPY docker/requirements.lock /opt/requirements.lock
RUN uv pip install -r /opt/requirements.lock

COPY env /opt/doom_env_src
RUN uv pip install /opt/doom_env_src

RUN useradd -m -u 1000 user
WORKDIR /workspace

COPY maps /workspace/maps
COPY submission /workspace/submission
COPY training /workspace/training
RUN chown -R 1000:1000 /workspace

USER user
ENV HOME=/home/user
EXPOSE 7860

CMD ["python", "-m", "training.web_demo", "--host", "0.0.0.0", "--port", "7860", "--device", "cpu"]
