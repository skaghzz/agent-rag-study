FROM mcr.microsoft.com/devcontainers/python:3.13

# uv, pyrefly, ruff 설치
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh \
    && pip install --upgrade pyrefly ruff \
    && uv --version \
    && pyrefly --version \
    && ruff --version