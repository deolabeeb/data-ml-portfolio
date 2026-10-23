import asyncio
import socket
import pytest
from fastapi.testclient import TestClient
from supportops.config import Settings
from supportops.main import create_app


@pytest.fixture
def test_event_loop():
    # Windows creates its internal socketpair using a loopback connect. Create
    # that infrastructure before forbidding every application connect call.
    loop = asyncio.new_event_loop()
    yield loop
    if not loop.is_closed():
        loop.close()


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch, test_event_loop):
    def blocked(*args, **kwargs):
        raise AssertionError("Network calls are forbidden in the local mock test suite.")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def client(test_event_loop):
    with TestClient(create_app(Settings()), backend_options={
        "loop_factory": lambda: test_event_loop,
    }) as test_client:
        yield test_client


@pytest.fixture
def auth_a():
    return {"Authorization": "Bearer demo-customer-a"}


@pytest.fixture
def auth_b():
    return {"Authorization": "Bearer demo-customer-b"}
