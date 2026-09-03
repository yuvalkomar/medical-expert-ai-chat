import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.llm.base import LLMProvider
from backend.app.main import create_app


def build_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        "log_file": str(tmp_path / "test.log"),
        "llm_provider": "mock",
        "max_concurrency": 2,
        "max_retries": 2,
        "retry_delay": 0.01,
        "shutdown_grace_period": 2,
        "mock_response_delay": 0,
    }
    values.update(overrides)
    return Settings(**values)


def wait_for_terminal(
    client: TestClient,
    message_id: str,
    *,
    timeout: float = 3,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/chat/{message_id}")
        assert response.status_code == 200
        last = response.json()
        if last["status"] in {"completed", "failed"}:
            return last
        time.sleep(0.01)
    pytest.fail(f"Message did not finish before timeout; last response: {last}")


@pytest.fixture
def client_factory(tmp_path: Path):
    def factory(
        provider: LLMProvider,
        *,
        database_name: str = "test.db",
        **setting_overrides: Any,
    ) -> TestClient:
        settings = build_settings(
            tmp_path,
            database_url=f"sqlite:///{(tmp_path / database_name).as_posix()}",
            **setting_overrides,
        )
        return TestClient(create_app(settings=settings, provider=provider))

    return factory

