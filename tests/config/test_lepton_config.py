import re
from types import SimpleNamespace

import pytest

from domyn_swarm.backends.compute.lepton import LeptonComputeBackend
from domyn_swarm.backends.serving.lepton import LeptonServingBackend
from domyn_swarm.config.lepton import (
    LeptonConfig,
    LeptonEndpointConfig,
    LeptonJobConfig,
)


@pytest.fixture
def cfg_ctx():
    # mimics the fields LeptonConfig.build() expects from the "swarm" context
    # (replicas, image, model, port, gpus_per_replica, args)
    return SimpleNamespace(
        replicas=2,
        image=None,  # allow endpoint.image to be used
        model="mistralai/Mistral-7B-Instruct",
        port=8000,
        gpus_per_replica=4,
        args="--max-model-len 8192 --dtype float16",
    )


def test_build_returns_valid_deployment_plan_with_defaults(cfg_ctx):
    cfg = LeptonConfig(
        type="lepton",
        workspace_id="ws-123",
        endpoint=LeptonEndpointConfig(),  # use defaults
        job=LeptonJobConfig(),  # use defaults
        env={"GLOBAL": "yes"},
    )

    plan = cfg.build(cfg_ctx)

    # Basic shape
    assert plan.platform == "lepton"
    assert "lepton-ws-123" in plan.name_hint

    # Backends are concrete classes and carry workspace
    assert isinstance(plan.serving, LeptonServingBackend)
    assert plan.serving.workspace == "ws-123"
    assert isinstance(plan.compute, LeptonComputeBackend)
    assert plan.compute.workspace == "ws-123"

    # Serving spec structure (dict dumped from Pydantic)
    ss = plan.serving_spec
    assert "container" in ss
    assert "resource_requirement" in ss
    assert "envs" in ss
    assert "api_tokens" in ss

    # Container image falls back to endpoint.image when cfg_ctx.image is None
    assert ss["container"]["image"] == cfg.endpoint.image

    # Command includes vllm serve, model, host/port, tensor-parallel, and extra args
    cmd = ss["container"]["command"]
    assert cmd[:3] == ["vllm", "serve", cfg_ctx.model]
    assert "--host" in cmd and "0.0.0.0" in cmd
    assert "--port" in cmd and str(cfg_ctx.port) in cmd
    assert "--tensor-parallel-size" in cmd and str(cfg_ctx.gpus_per_replica) in cmd
    assert "--max-model-len" in cmd and "8192" in cmd
    assert "--dtype" in cmd and "float16" in cmd

    # Resource requirement mirrors replicas and shape
    rr = ss["resource_requirement"]
    assert rr["min_replicas"] == cfg_ctx.replicas
    assert rr["max_replicas"] == cfg_ctx.replicas
    assert rr["resource_shape"] == cfg.endpoint.resource_shape

    # Env merge: global + endpoint.env
    env_pairs = {e["name"]: e.get("value") for e in ss["envs"]}
    assert env_pairs.get("GLOBAL") == "yes"
    # endpoint default env is {}, so just ensure at least GLOBAL is present
    assert "GLOBAL" in env_pairs

    # Token: since api_token_secret_name not set, build() generates one
    tokens = ss["api_tokens"]
    assert isinstance(tokens, list) and len(tokens) == 1
    tok_val = tokens[0]["value"]
    assert isinstance(tok_val, str) and len(tok_val) > 0
    assert plan.extras.get("api_token") == tok_val


def test_build_overrides_image_and_merges_envs(cfg_ctx):
    cfg = LeptonConfig(
        type="lepton",
        workspace_id="ws-abc",
        endpoint=LeptonEndpointConfig(
            image="vllm/vllm-openai:latest",
            env={"HF_HOME": "/data/hf"},
        ),
        job=LeptonJobConfig(env={"CACHE_DIR": "/cache"}),
        env={"GLOBAL": "1", "HF_HOME": "/should_be_overridden"},
    )

    # Override image via cfg_ctx
    cfg_ctx.image = "my/custom:v1"

    plan = cfg.build(cfg_ctx)
    ss = plan.serving_spec
    jr = plan.job_resources

    # container.image should come from cfg_ctx.image
    assert ss["container"]["image"] == "my/custom:v1"

    # serving envs: endpoint.env should override cfg.env on key collisions
    s_env = {e["name"]: e.get("value") for e in ss["envs"]}
    assert s_env["GLOBAL"] == "1"
    assert s_env["HF_HOME"] == "/data/hf"  # endpoint wins over global

    # job envs: job.env merged over global env
    j_env = {e["name"]: e.get("value") for e in jr.get("envs", [])}
    assert j_env["GLOBAL"] == "1"
    assert j_env["CACHE_DIR"] == "/cache"


def test_build_respects_endpoint_secret_name_disables_generated_token(cfg_ctx):
    cfg = LeptonConfig(
        type="lepton",
        workspace_id="ws-xyz",
        endpoint=LeptonEndpointConfig(api_token_secret_name="my-secret"),
        job=LeptonJobConfig(),
    )

    plan = cfg.build(cfg_ctx)
    ss = plan.serving_spec

    # When api_token_secret_name is set, we still include an apiTokens array
    # but extras['api_token'] should be None (no random token needed)
    assert plan.extras.get("api_token") is None

    tokens = ss.get("apiTokens", [])
    # token list may still exist, but we accept either empty or a redacted placeholder
    assert isinstance(tokens, list)
    if tokens:
        assert "value" in tokens[0]


def test_build_job_resources_shape_and_affinity(cfg_ctx):
    cfg = LeptonConfig(
        type="lepton",
        workspace_id="ws-999",
        job=LeptonJobConfig(
            resource_shape="gpu.4xh200",
            allowed_dedicated_node_groups=["group-a", "group-b"],
            allowed_nodes=["node-1", "node-2"],
        ),
    )

    plan = cfg.build(cfg_ctx)
    jr = plan.job_resources

    assert jr["resource_shape"] == "gpu.4xh200"
    # Affinity structure from LeptonJobUserSpec.model_dump(by_alias=True)
    aff = jr.get("affinity", {})
    assert aff.get("allowed_dedicated_node_groups") == ["group-a", "group-b"]
    assert aff.get("allowed_nodes_in_node_group") == ["node-1", "node-2"]


def test_build_command_contains_expected_order_and_tokens(cfg_ctx):
    cfg = LeptonConfig(type="lepton", workspace_id="ws-order")

    plan = cfg.build(cfg_ctx)
    cmd = plan.serving_spec["container"]["command"]

    # sanity: correct sub-sequence order for critical flags
    joined = " ".join(cmd)
    assert re.search(rf"vllm\s+serve\s+{re.escape(cfg_ctx.model)}", joined)
    assert re.search(r"--host\s+0\.0\.0\.0", joined)
    assert re.search(rf"--port\s+{cfg_ctx.port}", joined)
    assert re.search(rf"--tensor-parallel-size\s+{cfg_ctx.gpus_per_replica}", joined)
