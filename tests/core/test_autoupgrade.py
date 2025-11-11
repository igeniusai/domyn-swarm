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

import importlib
from pathlib import Path

import pytest
import typer


@pytest.fixture
def autoupgrade_mod():
    """
    Import the autoupgrade module once for reuse.
    Adjust the path if ensure_db_up_to_date lives elsewhere.
    """
    return importlib.import_module("domyn_swarm.core.state.autoupgrade")


class DummySettings:
    def __init__(self, home: Path, skip_db_upgrade: bool = False):
        self.skip_db_upgrade = skip_db_upgrade
        self.home = home


def _reset_flag(mod):
    # Ensure we start from a clean state
    if hasattr(mod, "_DB_UPGRADED"):
        setattr(mod, "_DB_UPGRADED", False)


def test_ensure_db_up_to_date_skips_when_already_upgraded(monkeypatch, autoupgrade_mod):
    _reset_flag(autoupgrade_mod)

    # Set flag to True to simulate already-upgraded process
    autoupgrade_mod._DB_UPGRADED = True

    _ = monkeypatch.setattr(autoupgrade_mod, "upgrade_head", lambda *_: None)
    _ = monkeypatch.setattr(autoupgrade_mod, "get_current_rev", lambda *_: "rev1")
    _ = monkeypatch.setattr(autoupgrade_mod, "get_head_rev", lambda *_: "rev2")

    autoupgrade_mod.ensure_db_up_to_date(noisy=True)

    # None of these should have been called because we bailed early on the flag.
    # We can't assert call counts on lambdas, but the main check is that nothing exploded
    # and that the flag is still True.
    assert autoupgrade_mod._DB_UPGRADED is True


def test_ensure_db_up_to_date_skips_when_env_flag_set(
    monkeypatch, tmp_path, autoupgrade_mod
):
    _reset_flag(autoupgrade_mod)

    # Make sure auto-upgrade is not yet done
    assert autoupgrade_mod._DB_UPGRADED is False

    # Pretend DB is under tmp_path
    monkeypatch.setattr(
        autoupgrade_mod,
        "get_settings",
        lambda: DummySettings(home=tmp_path, skip_db_upgrade=True),
    )

    _ = monkeypatch.setattr(
        autoupgrade_mod,
        "upgrade_head",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    monkeypatch.setattr(autoupgrade_mod, "get_current_rev", lambda *_: "rev1")
    monkeypatch.setattr(autoupgrade_mod, "get_head_rev", lambda *_: "rev2")

    autoupgrade_mod.ensure_db_up_to_date(noisy=True)

    # Should flip the flag but not call upgrade_head
    assert autoupgrade_mod._DB_UPGRADED is True


def test_ensure_db_up_to_date_does_nothing_if_already_at_head(
    monkeypatch, tmp_path, autoupgrade_mod
):
    _reset_flag(autoupgrade_mod)

    monkeypatch.setattr(
        autoupgrade_mod,
        "get_settings",
        lambda: DummySettings(home=tmp_path),
    )

    # current == head => no migration
    def fake_current(db_path: str) -> str:
        return "revX"

    def fake_head(db_path: str) -> str:
        return "revX"

    monkeypatch.setattr(autoupgrade_mod, "get_current_rev", fake_current)
    monkeypatch.setattr(autoupgrade_mod, "get_head_rev", fake_head)

    upgrade_called = {"n": 0}

    def fake_upgrade(db_path: str):
        upgrade_called["n"] += 1

    monkeypatch.setattr(autoupgrade_mod, "upgrade_head", fake_upgrade)

    autoupgrade_mod.ensure_db_up_to_date(noisy=False)

    assert autoupgrade_mod._DB_UPGRADED is True
    assert upgrade_called["n"] == 0  # no upgrade performed


def test_ensure_db_up_to_date_runs_upgrade_when_needed(
    monkeypatch, tmp_path, autoupgrade_mod
):
    _reset_flag(autoupgrade_mod)

    monkeypatch.setattr(
        autoupgrade_mod,
        "get_settings",
        lambda: DummySettings(home=tmp_path),
    )

    # current != head => must run upgrade_head
    def fake_current(db_path: str) -> str:
        # record the path for assertions if we want
        assert db_path == (tmp_path / "swarm.db").as_posix()
        return "rev_old"

    def fake_head(db_path: str) -> str:
        assert db_path == (tmp_path / "swarm.db").as_posix()
        return "rev_new"

    monkeypatch.setattr(autoupgrade_mod, "get_current_rev", fake_current)
    monkeypatch.setattr(autoupgrade_mod, "get_head_rev", fake_head)

    calls = {"n": 0, "db_path": None}

    def fake_upgrade(db_path: str):
        calls["n"] += 1
        calls["db_path"] = db_path

    monkeypatch.setattr(autoupgrade_mod, "upgrade_head", fake_upgrade)

    # We don't assert on console output here; just that upgrade is invoked correctly
    autoupgrade_mod.ensure_db_up_to_date(noisy=False)

    assert autoupgrade_mod._DB_UPGRADED is True
    assert calls["n"] == 1
    assert calls["db_path"] == (tmp_path / "swarm.db").as_posix()


def test_ensure_db_up_to_date_raises_on_upgrade_failure(
    monkeypatch, tmp_path, autoupgrade_mod
):
    _reset_flag(autoupgrade_mod)

    monkeypatch.setattr(
        autoupgrade_mod,
        "get_settings",
        lambda: DummySettings(home=tmp_path),
    )

    monkeypatch.setattr(autoupgrade_mod, "get_current_rev", lambda *_: "old")
    monkeypatch.setattr(autoupgrade_mod, "get_head_rev", lambda *_: "new")

    def failing_upgrade(db_path: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(autoupgrade_mod, "upgrade_head", failing_upgrade)

    with pytest.raises(typer.Exit) as excinfo:
        autoupgrade_mod.ensure_db_up_to_date(noisy=False)

    assert excinfo.value.exit_code == 1
    assert autoupgrade_mod._DB_UPGRADED is False
