# syntax=docker/dockerfile:1.7

############################
# Builder: resolve deps & build wheel
############################
ARG UV_BASE=ghcr.io/astral-sh/uv:python3.12-bookworm-slim
FROM ${UV_BASE} AS builder

# Faster installs & better caching in Docker
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --extra lepton
COPY . /app

# Build sdist + wheel into /src/dist/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build

############################
# Runtime: minimal image with just your app installed
############################
FROM python:3.12-slim-bookworm AS runtime

# Bring in the uv binary to install the wheel using uv (keeps final pip metadata consistent)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create non-root user
ENV APP_USER=app \
    APP_HOME=/app \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --home-dir ${APP_HOME} --shell /sbin/nologin ${APP_USER}

WORKDIR ${APP_HOME}

# Copy built wheel(s) from builder
COPY --from=builder /app/dist/ /tmp/dist/

# Install your package into the **system** environment, non-editable, compile bytecode
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --find-links /tmp/dist "domyn-swarm[lepton]" && \
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
