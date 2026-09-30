import os

os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import app as app_module
from src.llms.groqllm import GroqLLM


class FakeResponse:
    def __init__(self, content, additional_kwargs=None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = FakeResponse("# Fake Blog Title\n\nFake blog content for testing.")
    return llm


@pytest.fixture
def mock_research_llm():
    llm = MagicMock()
    llm.invoke.return_value = FakeResponse(
        "Fake research notes.",
        additional_kwargs={"executed_tools": [{"output": "Title: Example\nURL: https://example.com/source\nContent: ..."}]},
    )
    return llm


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-api-key"}


@pytest.fixture
def client(monkeypatch, mock_llm, mock_research_llm):
    """A TestClient with the database and LLM layers replaced by fakes,
    so hitting the real API never touches Postgres or Groq."""
    runs = {}

    async def fake_init_db():
        pass

    async def fake_close_db():
        pass

    async def fake_create_run(run_id, topic, language):
        runs[run_id] = {
            "run_id": run_id, "topic": topic, "language": language,
            "status": "pending", "blog_title": None, "blog_content": None,
        }

    async def fake_update_run_status(run_id, status, blog_title=None, blog_content=None):
        runs[run_id]["status"] = status
        if blog_title is not None:
            runs[run_id]["blog_title"] = blog_title
        if blog_content is not None:
            runs[run_id]["blog_content"] = blog_content

    async def fake_get_run(run_id):
        return runs.get(run_id)

    checkpointer = InMemorySaver()

    monkeypatch.setattr(app_module, "init_db", fake_init_db)
    monkeypatch.setattr(app_module, "close_db", fake_close_db)
    monkeypatch.setattr(app_module, "create_run", fake_create_run)
    monkeypatch.setattr(app_module, "update_run_status", fake_update_run_status)
    monkeypatch.setattr(app_module, "get_run", fake_get_run)
    monkeypatch.setattr(app_module, "get_checkpointer", lambda: checkpointer)

    monkeypatch.setattr(GroqLLM, "get_llm", lambda self: mock_llm)
    monkeypatch.setattr(GroqLLM, "get_research_llm", lambda self: mock_research_llm)

    with TestClient(app_module.app) as test_client:
        yield test_client
