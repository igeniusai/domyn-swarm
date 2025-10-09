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

from rich.console import Console
from rich.rule import Rule

from domyn_swarm.platform.protocols import ServingPhase, ServingStatus

from . import render_status


def main():
    console = Console()

    lepton_ok = ServingStatus(
        phase=ServingPhase.RUNNING,
        url="http://lepton-01.example:9000",
        detail={"raw_state": "Ready", "http": 200},
    )
    render_status(("research-lepton-01", "lepton", lepton_ok), console=console)

    console.print(Rule(style="dim"))

    slurm_init = ServingStatus(
        phase=ServingPhase.INITIALIZING,
        url=None,
        detail={"rep": "RUNNING", "lb": "RUNNING", "http": "unready"},
    )
    render_status(("batch-slurm-01", "slurm", slurm_init), console=console)

    console.print(Rule(style="dim"))

    slurm_pending = ServingStatus(
        phase=ServingPhase.PENDING, url=None, detail={"rep": "PENDING", "lb": "PENDING"}
    )
    lepton_failed = ServingStatus(
        phase=ServingPhase.FAILED,
        url="http://lepton-02.example:9000",
        detail={"raw_state": "Stopped", "http": 502},
    )
    render_status(
        [
            ("slurm-train-a", "slurm", slurm_pending),
            ("lepton-serve-a", "lepton", lepton_ok),
            ("lepton-serve-b", "lepton", lepton_failed),
        ],
        console=console,
    )


if __name__ == "__main__":
    main()
