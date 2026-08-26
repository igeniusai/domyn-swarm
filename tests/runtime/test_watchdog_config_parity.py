# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The YAML schema and the in-container runtime config must agree.

`runtime/watchdog.py` is bind-mounted into the vLLM container and run by that
container's interpreter, so it cannot import pydantic or anything from
`domyn_swarm` -- the two definitions genuinely have to be separate classes.

What must not differ is what they *say*: a field means the same thing under the
same name, and an option that fails to reach the process falls back to the value
the YAML schema advertises rather than a different one. These tests are the only
thing holding that line, so they enumerate the exceptions explicitly.
"""

from __future__ import annotations

import dataclasses

import pytest

from domyn_swarm.config.watchdog import (
    WatchdogConfig as YamlConfig,
    WatchdogRayConfig as YamlRayConfig,
)
from domyn_swarm.runtime.watchdog import (
    WatchdogConfig as RuntimeConfig,
    WatchdogRayConfig as RuntimeRayConfig,
)

# Declared in the YAML schema but deliberately absent from the runtime dataclass.
YAML_ONLY = {
    "enabled",  # gates whether the sbatch template launches the watchdog at all
    "ray",  # the nested model; compared separately
}

# Present only in the runtime dataclass: values injected per replica at launch,
# not user-configurable through YAML.
RUNTIME_ONLY = {
    "host",
    "port",
    "agent_version",
    "unhealthy_http_failures",
}

RUNTIME_ONLY_RAY = {
    "expected_workers",  # derived from SLURM_JOB_NUM_NODES, not from YAML
}


def _runtime_defaults(dc) -> dict:
    return {
        f.name: f.default for f in dataclasses.fields(dc) if f.default is not dataclasses.MISSING
    }


def _yaml_defaults(model) -> dict:
    return {name: field.default for name, field in model.model_fields.items()}


@pytest.mark.parametrize(
    ("yaml_model", "runtime_dc", "yaml_only", "runtime_only"),
    [
        (YamlConfig, RuntimeConfig, YAML_ONLY, RUNTIME_ONLY),
        (YamlRayConfig, RuntimeRayConfig, set(), RUNTIME_ONLY_RAY),
    ],
    ids=["WatchdogConfig", "WatchdogRayConfig"],
)
def test_field_names_match(yaml_model, runtime_dc, yaml_only, runtime_only):
    """A setting is spelled the same on both sides of the process boundary."""
    yaml_names = set(_yaml_defaults(yaml_model)) - yaml_only
    runtime_names = set(_runtime_defaults(runtime_dc)) - runtime_only

    missing_in_runtime = sorted(yaml_names - runtime_names)
    missing_in_yaml = sorted(runtime_names - yaml_names)

    assert not missing_in_runtime, (
        f"YAML fields with no runtime counterpart: {missing_in_runtime}. "
        f"Either add them, or list them in YAML_ONLY with a reason."
    )
    assert not missing_in_yaml, (
        f"Runtime fields with no YAML counterpart: {missing_in_yaml}. "
        f"Either add them, or list them in RUNTIME_ONLY with a reason."
    )


@pytest.mark.parametrize(
    ("yaml_model", "runtime_dc", "yaml_only"),
    [
        (YamlConfig, RuntimeConfig, YAML_ONLY),
        (YamlRayConfig, RuntimeRayConfig, set()),
    ],
    ids=["WatchdogConfig", "WatchdogRayConfig"],
)
def test_defaults_match(yaml_model, runtime_dc, yaml_only):
    """An option that fails to reach the process falls back to the documented value."""
    yaml_defaults = _yaml_defaults(yaml_model)
    runtime_defaults = _runtime_defaults(runtime_dc)

    mismatched = {
        name: (value, runtime_defaults[name])
        for name, value in yaml_defaults.items()
        if name not in yaml_only and name in runtime_defaults and value != runtime_defaults[name]
    }

    assert not mismatched, (
        "Same field, different default (yaml, runtime): "
        f"{mismatched}. A field that is not forwarded would silently behave "
        f"differently from what the YAML schema documents."
    )


# The argparse layer is what actually governs when a flag is omitted, so it has to
# agree with the YAML schema too. Maps argparse dest -> YAML field name.
ARGV_TO_YAML = {
    "http_path": "http_path",
    "http_timeout": "http_timeout",
    "probe_interval": "probe_interval",
    "readiness_timeout": "readiness_timeout",
    "unhealthy_restart_after": "unhealthy_restart_after",
    "restart_policy": "restart_policy",
    "restart_backoff": "restart_backoff_initial",
    "restart_backoff_max": "restart_backoff_max",
    "kill_grace_seconds": "kill_grace_seconds",
    "max_restarts": "max_restarts",
    "log_level": "log_level",
}

ARGV_TO_YAML_RAY = {
    "ray_timeout": "probe_timeout_s",
    "ray_grace": "status_grace_s",
    "ray_probe_interval": "probe_interval_s",
}

_MINIMAL_ARGV = [
    "--swarm-id",
    "s",
    "--replica-id",
    "0",
    "--node",
    "n",
    "--port",
    "8000",
    "--log-dir",
    "/tmp",
    "--collector-address",
    "127.0.0.1:9100",
]


@pytest.mark.parametrize(
    ("mapping", "yaml_model"),
    [(ARGV_TO_YAML, YamlConfig), (ARGV_TO_YAML_RAY, YamlRayConfig)],
    ids=["WatchdogConfig", "WatchdogRayConfig"],
)
def test_argparse_defaults_match_the_yaml_schema(mapping, yaml_model):
    """Omitting a flag must land on the value the YAML schema documents."""
    from domyn_swarm.runtime.watchdog import _parse_args

    args = _parse_args(list(_MINIMAL_ARGV))
    yaml_defaults = _yaml_defaults(yaml_model)

    mismatched = {
        dest: (getattr(args, dest), yaml_defaults[yaml_name])
        for dest, yaml_name in mapping.items()
        if getattr(args, dest) != yaml_defaults[yaml_name]
    }

    assert not mismatched, (
        f"argparse default differs from the YAML default (argv, yaml): {mismatched}"
    )


def test_every_yaml_field_is_covered_by_one_of_these_checks():
    """A new YAML field cannot be added without deciding how it reaches the process."""
    covered = set(ARGV_TO_YAML.values()) | YAML_ONLY
    uncovered = sorted(set(YamlConfig.model_fields) - covered)

    assert not uncovered, (
        f"WatchdogConfig fields not mapped to a watchdog argument: {uncovered}. "
        f"Add them to ARGV_TO_YAML, or to YAML_ONLY with a reason."
    )
