import pytest
from domyn_swarm.jobs.clients.factory import create_llm_client
from domyn_swarm.jobs.clients.openai import OpenAIClient


def test_create_openai_client():
    client = create_llm_client("openai", "http://dummy-endpoint", timeout=123)
    assert isinstance(client, OpenAIClient)


def test_unknown_client_provider():
    with pytest.raises(ValueError, match="Unsupported provider: foo"):
        create_llm_client("foo", "http://localhost")
