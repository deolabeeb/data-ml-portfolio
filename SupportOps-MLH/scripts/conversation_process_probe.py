"""Synthetic API probe run in separate terminated processes by S02 tests."""
import asyncio
import json
import os
from pathlib import Path
import socket
import sys
from fastapi.testclient import TestClient
from supportops.config import Settings
from supportops.main import create_app

loop = asyncio.new_event_loop()  # Windows infrastructure before outbound guard.
def blocked(*args, **kwargs):
    raise AssertionError("Network forbidden in process probe")
socket.create_connection = blocked
socket.socket.connect = blocked
with TestClient(create_app(Settings(conversation_db=Path(sys.argv[1]))),
                backend_options={"loop_factory": lambda: loop}) as client:
    a = {"Authorization": "Bearer demo-customer-a"}
    b = {"Authorization": "Bearer demo-customer-b"}
    if sys.argv[2] == "create":
        created = client.post("/v1/conversations", json={}, headers=a)
        assert created.status_code == 201
        cid = created.json()["conversation_id"]
        message = "Where is ORD-1001?"
    else:
        cid = sys.argv[3]
        message = "And when will it arrive?"
    result = client.post(f"/v1/conversations/{cid}/chat", json={"message": message}, headers=a)
    assert result.status_code == 200, result.text
    foreign = client.post(f"/v1/conversations/{cid}/chat", json={"message": message}, headers=b)
    assert foreign.status_code == 404
    print(json.dumps({**result.json(), "pid": os.getpid(), "foreign_status": foreign.status_code}))
if not loop.is_closed():
    loop.close()
