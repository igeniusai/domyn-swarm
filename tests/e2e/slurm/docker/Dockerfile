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

FROM base AS vllm
RUN dnf install -y numactl-libs && dnf clean all
# 0.30.0 and later wheels need glibc 2.39; openEuler 24.03 ships 2.38.
ARG VLLM_VERSION=0.29.0
ARG TARGETARCH
RUN case "$TARGETARCH" in \
      amd64) arch=x86_64 ;; \
      arm64) arch=aarch64 ;; \
      *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && uv venv /opt/vllm \
    && uv pip install --python /opt/vllm/bin/python --torch-backend cpu \
      "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_${arch}.whl" \
    && rm -rf /root/.cache/uv
COPY --chmod=755 vllm-cpu.sh /usr/local/bin/vllm
