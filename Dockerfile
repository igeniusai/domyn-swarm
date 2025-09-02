# syntax=docker/dockerfile:1.7

############################
# Builder: resolve deps & build wheel
############################
ARG UV_BASE=ghcr.io/astral-sh/uv:python3.12-bookworm-slim
FROM ${UV_BASE} AS builder

# Faster installs & better caching in Docker
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# (Optional) OS build deps for native wheels; add as needed
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# 1) Copy only files that affect dependency resolution first → better layer cache
COPY pyproject.toml uv.lock* README* LICENSE* ./

# Pre-populate uv’s cache so later steps are fast (no project source yet)
# Uses BuildKit cache mount; requires the `# syntax=` line above.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml

# 2) Now add the rest of the source and build distribution artifacts
COPY . .

# Build sdist + wheel into /src/dist/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build

############################
# Runtime: minimal image with just your app installed
############################
FROM python:3.12-slim-bookworm AS runtime

# Bring in the uv binary to install the wheel using uv (keeps final pip metadata consistent)
COPY --from=builder /uv /uvx /bin/

# Create non-root user
ENV APP_USER=app \
    APP_HOME=/app \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --home-dir ${APP_HOME} --shell /sbin/nologin ${APP_USER}

WORKDIR ${APP_HOME}

# Copy built wheel(s) from builder
COPY --from=builder /src/dist/ /tmp/dist/

# Install your package into the **system** environment, non-editable, compile bytecode
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system /tmp/dist/*.whl && \
    rm -rf /tmp/dist

# Drop privileges
USER ${APP_USER}

# If your package exposes a console_script (recommended),
# set the entrypoint to it. Replace "your-cli" with your actual command.
ENTRYPOINT ["domyn-swarm"]
# Provide a default arg so `docker run image` shows help
CMD ["--help"]

# OCI labels (optional but nice)
LABEL org.opencontainers.image.title="domyn-swarm" \
      org.opencontainers.image.description="" \
