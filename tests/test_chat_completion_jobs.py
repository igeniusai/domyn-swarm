from unittest.mock import AsyncMock

import pandas as pd
import pytest

from domyn_swarm.jobs import (
    ChatCompletionJob,
    ChatCompletionPerplexityJob,
    CompletionJob,
    MultiChatCompletionJob,
    MultiTurnChatCompletionJob,
)


@pytest.mark.asyncio
async def test_completion_job(monkeypatch, tmp_path):
    df = pd.DataFrame({"prompt": ["Hello", "World"]})

    mock_resp = AsyncMock()
    mock_resp.choices = [type("C", (), {"text": "Mocked completion"})()]

    job = CompletionJob(model="gpt-3")
    job.client = AsyncMock()
    job.client.completions.create = AsyncMock(return_value=mock_resp)

    results = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    assert len(results) == 2
    print(results)
    assert results["completion"][0] == "Mocked completion"


@pytest.mark.asyncio
async def test_chat_completion_job(monkeypatch, tmp_path):
    df = pd.DataFrame({"messages": [[{"role": "user", "content": "Hi"}]]})

    mock_choice = type(
        "Choice", (), {"message": type("Msg", (), {"content": "Mocked answer"})()}
    )
    mock_resp = AsyncMock()
    mock_resp.choices = [mock_choice]

    job = ChatCompletionJob(model="gpt-4")
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    results = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    assert len(results) == 1
    assert results["result"][0] == "Mocked answer"


@pytest.mark.asyncio
async def test_multi_chat_completion_job(monkeypatch, tmp_path):
    df = pd.DataFrame({"messages": [[{"role": "user", "content": "Hi"}]]})

    mock_choices = [
        type("Choice", (), {"message": type("Msg", (), {"content": f"Mock {i}"})()})
        for i in range(3)
    ]
    mock_resp = AsyncMock()
    mock_resp.choices = mock_choices

    job = MultiChatCompletionJob(model="gpt-4", n=3, output_column_name="generated")
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    _ = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    print(job.results)
    assert job.results["generated_1"][0] == "Mock 0"
    assert job.results["generated_2"][0] == "Mock 1"
    assert job.results["generated_3"][0] == "Mock 2"


@pytest.mark.asyncio
async def test_chat_completion_perplexity_job(monkeypatch, tmp_path):
    df = pd.DataFrame({"messages": [[{"role": "user", "content": "Hi"}]]})

    mock_choice = type(
        "Choice",
        (),
        {
            "message": type("Msg", (), {"content": "text!"})(),
            "logprobs": {"tokens": [], "token_logprobs": []},
        },
    )
    mock_resp = AsyncMock()
    mock_resp.choices = [mock_choice]

    monkeypatch.setattr(
        "domyn_swarm.jobs.chat_completion.extract_token_logprobs",
        lambda c: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        "domyn_swarm.jobs.chat_completion.compute_perplexity_metrics",
        lambda t: (42.0, 21.0),
    )

    job = ChatCompletionPerplexityJob(model="gpt-4")
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    _ = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    print(job.results.columns)
    assert job.results["text"][0] == "text!"
    assert job.results["perplexity"][0] == 42.0
    assert job.results["bottom50_perplexity"][0] == 21.0


@pytest.mark.asyncio
async def test_multi_turn_chat_completion_job(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "messages": [
                [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Tell me a joke"},
                ]
            ]
        }
    )

    mock_resp = AsyncMock()
    mock_resp.choices = [
        type("C", (), {"message": type("M", (), {"content": "Mocked reply"})()})
    ]

    job = MultiTurnChatCompletionJob(model="gpt-4")
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    _ = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    print(job.results)
    assert isinstance(job.results["results"][0], list)
    assert any(m.get("content") == "Mocked reply" for m in job.results["results"][0])
