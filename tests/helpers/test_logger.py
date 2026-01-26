import logging

from domyn_swarm.helpers.logger import setup_logger


def test_setup_logger_adds_handlers(monkeypatch):
    """Creates handlers and respects stderr routing.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    logger = setup_logger(name="domyn_swarm.test_logger", level=logging.INFO, to_stderr=True)
    assert logger.handlers
    assert any(getattr(h, "level", None) == logging.INFO for h in logger.handlers)


def test_setup_logger_idempotent():
    """Returns existing logger when handlers are already configured."""
    name = "domyn_swarm.test_logger.idempotent"
    logger_first = setup_logger(name=name, level=logging.INFO)
    handler_count = len(logger_first.handlers)
    logger_second = setup_logger(name=name, level=logging.DEBUG)
    assert logger_second is logger_first
    assert len(logger_second.handlers) == handler_count
