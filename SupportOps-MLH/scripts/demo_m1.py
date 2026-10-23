"""Interactive, synthetic two-customer API demo and controlled failure path.

Uses FastAPI TestClient, not a deployed HTTP service. No network or model calls.
Choose a fresh output folder; actual commands/responses are saved after exit.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

from fastapi.testclient import TestClient
from supportops.config import Settings
from supportops.execution import ExecutionPolicy
from supportops.main import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    db = out / "demo.sqlite3"
    loop, failure_loop = asyncio.new_event_loop(), asyncio.new_event_loop()
    def blocked(*args, **kwargs):
        raise AssertionError("Network forbidden in M1 demo")
    socket.create_connection = blocked
    socket.socket.connect = blocked
    headers = {"a": {"Authorization": "Bearer demo-customer-a"},
               "b": {"Authorization": "Bearer demo-customer-b"}}
    events = []
    cid = None
    print("Local synthetic M1 demo; no network, model calls, or actions.")
    print("1 create A | 2 A order | 3 A follow-up | 4 B denied | 5 bounded failure | 6 A recover | q quit")
    with TestClient(create_app(Settings(conversation_db=db)), backend_options={"loop_factory": lambda: loop}) as normal, \
         TestClient(create_app(Settings(conversation_db=db, execution=ExecutionPolicy(max_steps=1))),
                    backend_options={"loop_factory": lambda: failure_loop}) as limited:
        while True:
            choice = input("Choice: ").strip().lower()
            if choice == "q":
                break
            if choice == "1":
                response = normal.post("/v1/conversations", json={}, headers=headers["a"])
                cid = response.json()["conversation_id"]
            elif choice in {"2", "3", "4", "5", "6"} and cid:
                client = limited if choice == "5" else normal
                message = "Where is ORD-1001?" if choice == "2" else "And when will it arrive?"
                response = client.post(f"/v1/conversations/{cid}/chat", json={"message": message},
                                       headers=headers["b" if choice == "4" else "a"])
            else:
                print("Create a conversation first; choose 1–6 or q.")
                continue
            event = {"choice": choice, "status": response.status_code, "response": response.json()}
            events.append(event)
            print(json.dumps(event))
    for event_loop in (loop, failure_loop):
        if not event_loop.is_closed():
            event_loop.close()
    result = {"completed_at": datetime.now(timezone.utc).isoformat(), "mode": "mock", "llm_calls": 0,
              "transport": "in-process FastAPI TestClient", "events": events}
    (out / "demo.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out / 'demo.json'}")


if __name__ == "__main__":
    main()
