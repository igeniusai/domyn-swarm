# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from domyn_swarm.config.swarm import DomynLLMSwarmConfig


def _cfg(*, gpus_per_replica, gpus_per_node, replicas, backend_extra=None):
    """Build a validated swarm config with a fresh backend dict each call.

    The validator mutates the backend dict (e.g. sets ``requires_ray``), so
    every case must use its own dict to avoid cross-test leakage.
    """
    backend = {"type": "slurm", "account": "a", "partition": "p", "qos": "q"}
    if backend_extra:
        backend.update(backend_extra)
    return DomynLLMSwarmConfig.model_validate(
        {
            "name": "s",
            "model": "m",
            "image": "v.sif",
            "replicas": replicas,
            "gpus_per_replica": gpus_per_replica,
            "gpus_per_node": gpus_per_node,
            "backend": backend,
        }
    )


def test_ray_config_selects_ray_template():
    cfg = _cfg(gpus_per_replica=8, gpus_per_node=4, replicas=1)
    assert cfg.backend.requires_ray is True
    assert cfg.backend.template_path.name == "llm_swarm_ray.sh.j2"


def test_non_ray_config_keeps_default_template():
    cfg = _cfg(gpus_per_replica=1, gpus_per_node=4, replicas=2)
    assert cfg.backend.requires_ray is False
    assert cfg.backend.template_path.name == "llm_swarm.sh.j2"


def test_user_template_override_respected_for_ray():
    cfg = _cfg(
        gpus_per_replica=8,
        gpus_per_node=4,
        replicas=1,
        backend_extra={"template_path": "/custom/my.sh.j2"},
    )
    assert cfg.backend.requires_ray is True
    assert cfg.backend.template_path.name == "my.sh.j2"


def _ray_cfg(**mon):
    """Build a validated swarm config with a fresh backend dict, requiring Ray.

    Mirrors ``_cfg`` above: the validator mutates the backend dict, so each
    case gets its own fresh dict to avoid cross-test leakage.
    """
    from domyn_swarm.config.swarm import DomynLLMSwarmConfig

    return DomynLLMSwarmConfig.model_validate(
        {
            "name": "s",
            "model": "m",
            "image": "v.sif",
            "replicas": 1,
            "gpus_per_replica": 8,
            "gpus_per_node": 4,
            "backend": {
                "type": "slurm",
                "account": "a",
                "partition": "p",
                "qos": "q",
                "endpoint": {"monitoring": {"enabled": True, "mode": "binary", **mon}},
            },
        }
    )


def test_ray_metrics_auto_on_when_monitoring_and_ray():
    cfg = _ray_cfg()
    assert cfg.backend.endpoint.monitoring.ray_metrics.enabled is True


def test_ray_metrics_explicit_false_respected():
    cfg = _ray_cfg(ray_metrics={"enabled": False})
    assert cfg.backend.endpoint.monitoring.ray_metrics.enabled is False


def test_ray_metrics_off_when_not_ray():
    from domyn_swarm.config.swarm import DomynLLMSwarmConfig

    cfg = DomynLLMSwarmConfig.model_validate(
        {
            "name": "s",
            "model": "m",
            "image": "v.sif",
            "replicas": 2,
            "gpus_per_replica": 1,
            "gpus_per_node": 4,
            "backend": {
                "type": "slurm",
                "account": "a",
                "partition": "p",
                "qos": "q",
                "endpoint": {"monitoring": {"enabled": True, "mode": "binary"}},
            },
        }
    )
    assert cfg.backend.endpoint.monitoring.ray_metrics.enabled is False
