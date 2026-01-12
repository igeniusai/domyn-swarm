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

from types import SimpleNamespace

from domyn_swarm.config.lepton import LeptonConfig
from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig


class SlurmCfgCtx:
    def __init__(self):
        self.replicas = 2
        self.nodes = 1
        self.gpus_per_replica = 1
        self.gpus_per_node = 4
        self.replicas_per_node = 2

    def model_dump(self, *, include=None, exclude_none=False):
        data = {
            "replicas": self.replicas,
            "nodes": self.nodes,
            "gpus_per_replica": self.gpus_per_replica,
            "gpus_per_node": self.gpus_per_node,
            "replicas_per_node": self.replicas_per_node,
        }
        if include is not None:
            data = {k: v for k, v in data.items() if k in include}
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


def test_slurm_plan_output_consistency():
    cfg = SlurmConfig(
        partition="debug",
        account="acct",
        qos="normal",
        endpoint=SlurmEndpointConfig(nginx_image="nginx:latest"),
    )
    plan = cfg.build(SlurmCfgCtx())

    assert plan.platform == "slurm"
    assert isinstance(plan.serving_spec, dict)
    assert isinstance(plan.job_resources, dict)
    assert isinstance(plan.extras, dict)
    assert isinstance(plan.shared_env, dict)
    assert plan.serving_spec["replicas"] == 2
    assert plan.serving_spec["gpus_per_replica"] == 1


def test_lepton_plan_output_consistency():
    cfg_ctx = SimpleNamespace(
        replicas=1,
        image=None,
        model="mistral",
        port=8000,
        gpus_per_replica=1,
        args="",
    )
    cfg = LeptonConfig(type="lepton", workspace_id="ws-123")
    plan = cfg.build(cfg_ctx)

    assert plan.platform == "lepton"
    assert isinstance(plan.serving_spec, dict)
    assert isinstance(plan.job_resources, dict)
    assert isinstance(plan.extras, dict)
    assert isinstance(plan.shared_env, dict)
    assert plan.extras.get("workspace_id") == "ws-123"
