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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from domyn_swarm.config.lepton import LeptonConfig
from domyn_swarm.config.slurm import SlurmConfig
from domyn_swarm.exceptions import JobNotFoundError
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.platform.protocols import ServingHandle

if TYPE_CHECKING:
    from domyn_swarm.backends.compute.slurm import SlurmComputeBackend

    from ..swarm import DomynLLMSwarm

import dataclasses

from domyn_swarm.config.settings import get_settings

from .db import make_session_factory
from .models import SwarmRecord

logger = setup_logger(__name__)


def _escape_like(s: str) -> str:
    # Escape %, _ and \ for SQLite LIKE ... ESCAPE '\'
    return s.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


class SwarmStateManager:
    DB_NAME = "swarm.db"

    def __init__(self, swarm: "DomynLLMSwarm"):
        self.swarm = swarm

    # --- path helpers ---
    @classmethod
    def _get_db_path(cls) -> Path:
        return get_settings().home / cls.DB_NAME

    # --- CRUD ---
    def save(self, deployment_name: str) -> None:
        session_factory = make_session_factory(self._get_db_path())
        # Build payload
        cfg_obj = self.swarm.cfg
        swarm_obj = self.swarm.model_dump(
            mode="json", exclude={"cfg": True}, by_alias=True, exclude_none=True
        )
        handle = self.swarm.serving_handle
        if handle is None:
            raise ValueError("Null Serving handle")

        handle_dict = dataclasses.asdict(handle)

        rec = SwarmRecord(
            deployment_name=deployment_name,
            swarm=swarm_obj,
            cfg=cfg_obj.model_dump(mode="json", by_alias=True, exclude_none=True),
            serving_handle=handle_dict,
        )
        with session_factory() as s:
            # UPSERT semantics: merge by PK
            s.merge(rec)
            s.commit()
        logger.debug(f"State saved for swarm {deployment_name}.")

    @classmethod
    def load(cls, deployment_name: str):
        from ... import DomynLLMSwarm  # avoid import cycles

        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rec = s.get(SwarmRecord, deployment_name)
        if rec is None:
            raise JobNotFoundError(deployment_name)

        # Rehydrate
        swarm_dict = rec.swarm
        swarm_dict["cfg"] = rec.cfg
        handle_dict = rec.serving_handle

        swarm = DomynLLMSwarm.model_validate(swarm_dict)
        serving_handle = ServingHandle(**handle_dict)

        swarm.serving_handle = serving_handle
        swarm._deployment._handle = serving_handle

        platform = swarm._platform
        if platform == "slurm":
            backend = cls._get_slurm_backend(
                handle=swarm.serving_handle, slurm_cfg=swarm.cfg.backend
            )
        elif platform == "lepton":
            from domyn_swarm.backends.compute.lepton import LeptonComputeBackend

            assert isinstance(swarm.cfg.backend, LeptonConfig)
            backend = LeptonComputeBackend(
                workspace=swarm.cfg.backend.workspace_id
            )  # adjust as needed
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        swarm._deployment.compute = backend
        return swarm

    def delete_record(self, deployment_name: str) -> None:
        session_factory = make_session_factory(self._get_db_path())
        with session_factory() as s:
            rec = s.get(SwarmRecord, deployment_name)
            if rec:
                s.delete(rec)
                s.commit()

    @classmethod
    def list_all(cls) -> list[dict[str, Any]]:
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rows = s.query(SwarmRecord).order_by(SwarmRecord.creation_dt.desc()).all()
        return [
            {
                "deployment_name": r.deployment_name,
                "swarm": r.swarm,
                "cfg": r.cfg,
                "serving_handle": r.serving_handle,
                "creation_dt": r.creation_dt.isoformat(),
                # Optional: convenience fan-out of commonly used fields:
                "endpoint": (r.swarm or {}).get("endpoint", ""),
                "platform": (r.swarm or {}).get("_platform", ""),
                "name": (r.swarm or {}).get("name", r.deployment_name),
            }
            for r in rows
        ]

    @classmethod
    def iter_all(cls) -> Iterable[dict[str, Any]]:
        for rec in cls.list_all():
            yield rec

    @classmethod
    def _get_slurm_backend(cls, handle, slurm_cfg) -> "SlurmComputeBackend":
        from domyn_swarm.backends.compute.slurm import SlurmComputeBackend

        jobid = handle.meta.get("jobid")
        lb_jobid = handle.meta.get("lb_jobid")
        lb_node = handle.meta.get("lb_node")
        if jobid is None:
            raise ValueError("State file does not contain valid job IDs")
        if lb_jobid is None or lb_node is None:
            raise ValueError("State file does not contain valid LB job info")
        assert isinstance(slurm_cfg, SlurmConfig)
        return SlurmComputeBackend(cfg=slurm_cfg, lb_jobid=lb_jobid, lb_node=lb_node)

    @classmethod
    def get_last_swarm_name(cls) -> str | None:
        """Get the last deployment name.

        Returns:
            str | None: Deployment name or None if no deployments exist.
        """
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rec = s.query(SwarmRecord).order_by(SwarmRecord.creation_dt.desc()).first()
        if rec is None:
            return None
        return rec.deployment_name

    @classmethod
    def list_by_base_name(cls, base_name: str) -> list[str]:
        pattern = f"{_escape_like(base_name)}-%"
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rows = (
                s.query(SwarmRecord)
                .filter(SwarmRecord.deployment_name.like(pattern, escape="\\"))
                .order_by(SwarmRecord.creation_dt.desc())
                .all()
            )
        return [r.deployment_name for r in rows]
