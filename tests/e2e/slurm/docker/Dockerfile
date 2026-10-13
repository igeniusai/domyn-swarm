# syntax=docker/dockerfile:1
ARG SLURM_TAG=26.05.4.1-oe2403sp4

FROM openeuler/slurm:${SLURM_TAG} AS base
RUN dnf install -y python3 nginx curl procps-ng hostname findutils \
    && dnf clean all
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
COPY slurm.conf gres.conf cgroup.conf /etc/slurm/
COPY --chmod=755 entrypoint.sh /usr/local/bin/e2e-entrypoint
ENV UV_PYTHON=/usr/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/shared/venv \
    UV_LINK_MODE=copy
ENTRYPOINT ["/usr/local/bin/e2e-entrypoint"]

FROM base AS fast
RUN ln -s /workspace/tests/e2e/slurm/shims/fake-vllm/vllm /usr/local/bin/vllm
