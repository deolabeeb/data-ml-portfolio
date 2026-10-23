"""S03 deterministic budget/failure tests, no external I/O or paid calls."""
import json
import logging
from threading import Event
from time import monotonic
from uuid import UUID

from fastapi.testclient import TestClient
from langgraph.graph import StateGraph, START
import pytest

from supportops.config import Settings
from supportops.conversations import ConversationService
from supportops.execution import ExecutionBudget, ExecutionPolicy, ExecutionFailure, ReadOnlyRunner
from supportops.graph import GraphSupportProvider
from supportops.main import create_app
from supportops.provider import MockSupportProvider
from supportops.schemas import Principal

A = Principal(customer_id="cust_demo_a")


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now
    def advance(self, seconds):
        self.now += seconds


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"max_steps": 33}, {"max_tools": True},
                                     {"max_requests": -1}, {"tool_seconds": 0},
                                     {"turn_seconds": 21}, {"turn_seconds": float("nan")}])
def test_invalid_limits_fail_closed(kwargs):
    with pytest.raises(ValueError):
        ExecutionPolicy(**kwargs)


@pytest.mark.parametrize("limit,outcome", [("max_steps", "step_limit"),
                                          ("max_requests", "request_limit"), ("max_tools", "tool_limit")])
def test_counters_prevent_next_operation(limit, outcome):
    budget = ExecutionBudget(ExecutionPolicy(**{limit: 1}), runner=ReadOnlyRunner())
    calls = []
    with pytest.raises(ExecutionFailure) as failure:
        if limit == "max_steps":
            budget.step()
            budget.step()
        else:
            budget.call("policy", lambda: calls.append(1) or "ok")
            budget.call("policy", lambda: calls.append(2) or "ok")
    assert failure.value.outcome == outcome
    assert calls == ([] if limit == "max_steps" else [1])


@pytest.mark.parametrize("seconds,tool_seconds,turn_seconds,outcome", [
    (2, 1, 20, "tool_timeout"), (20, 2, 20, "turn_timeout"), (1, 1, 20, "tool_timeout")])
def test_fake_clock_deadline_boundaries(seconds, tool_seconds, turn_seconds, outcome):
    clock = FakeClock()
    budget = ExecutionBudget(ExecutionPolicy(tool_seconds=tool_seconds, turn_seconds=turn_seconds), clock=clock)
    def slow():
        clock.advance(seconds)
        return "ok"
    with pytest.raises(ExecutionFailure) as failure:
        budget.call("policy", slow)
    assert failure.value.outcome == outcome


def test_no_dispatch_after_turn_expired():
    clock = FakeClock()
    budget = ExecutionBudget(clock=clock)
    clock.advance(20)
    with pytest.raises(ExecutionFailure, match="turn_timeout"):
        budget.call("policy", lambda: pytest.fail("expired operation dispatched"))


def test_real_slow_worker_timeout_capacity_and_cleanup():
    entered, release, exited = Event(), Event(), Event()
    runner = ReadOnlyRunner(capacity=1)
    budget = ExecutionBudget(ExecutionPolicy(tool_seconds=.05), runner=runner)
    def blocking_read():
        entered.set()
        try:
            release.wait(5)  # Bounded test cleanup, not an indefinitely hanging test.
            return "late result"
        finally:
            exited.set()
    started = monotonic()
    try:
        with pytest.raises(ExecutionFailure, match="tool_timeout"):
            budget.call("policy", blocking_read)
        assert entered.is_set() and monotonic() - started < 1
        assert not exited.is_set()  # Timeout does NOT claim the worker was killed.
        with pytest.raises(ExecutionFailure, match="capacity_exhausted"):
            runner.run(lambda: pytest.fail("exhausted runner dispatched"), .05)
    finally:
        release.set()
        assert exited.wait(1)
    assert runner.slots.acquire(timeout=1)
    runner.slots.release()
    assert runner.run(lambda: "recovered", 1) == "recovered"


def test_loop_hits_actual_langgraph_recursion_limit():
    provider = GraphSupportProvider(policy=ExecutionPolicy(max_steps=3))
    calls = []
    builder = StateGraph(dict)
    builder.add_node("loop", lambda state: calls.append(1) or state)
    builder.add_edge(START, "loop")
    builder.add_edge("loop", "loop")
    provider.graph = builder.compile()
    with pytest.raises(ExecutionFailure, match="step_limit") as failure:
        provider.respond("hello", A)
    assert failure.value.outcome == "step_limit"
    assert len(calls) == 3


@pytest.mark.parametrize("kind,outcome", [("fail", "tool_failure"), ("malformed", "malformed_output"),
                                         ("noop", "no_op"), ("timeout", "tool_timeout")])
def test_api_explicit_safe_failure(kind, outcome, test_event_loop):
    class Broken(MockSupportProvider):
        def order_status(self, plan, principal):
            if kind == "fail":
                raise RuntimeError("secret-token raw message hidden reasoning")
            if kind == "timeout":
                raise TimeoutError("private provider trace")
            if kind == "malformed":
                return {"intent": "shell", "answer": "secret"}
            return None
    provider = GraphSupportProvider(Broken())
    with TestClient(create_app(provider=provider), backend_options={"loop_factory": lambda: test_event_loop}) as client:
        result = client.post("/v1/chat", json={"message": "ORD-1001"},
                             headers={"Authorization": "Bearer demo-customer-a"})
    assert result.status_code == (504 if kind == "timeout" else 503)
    detail = result.json()["detail"]
    assert detail["outcome"] == outcome
    assert UUID(detail["request_id"])
    assert detail["mode"] == "mock" and detail["llm_calls"] == 0 and detail["action_executed"] is False
    assert "secret" not in result.text and "private" not in result.text


def test_logs_correlate_without_untrusted_data(caplog):
    caplog.set_level(logging.INFO, logger="supportops.execution")
    provider = GraphSupportProvider()
    response = provider.respond("Return policy? secret-token hidden reasoning", A)
    records = [json.loads(r.message) for r in caplog.records if r.name == "supportops.execution"]
    assert len(records) == 1
    assert records[0]["request_id"] == str(response.request_id)
    assert records[0]["outcome"] == "policy" and records[0]["steps"] == 3
    assert set(records[0]) == {"request_id", "mode", "outcome", "operation", "steps", "requests",
                               "tools", "elapsed_ms", "llm_calls", "retries"}
    assert "secret-token" not in caplog.text and "hidden reasoning" not in caplog.text
    assert "cust_demo_a" not in caplog.text and "Return policy" not in caplog.text


@pytest.mark.parametrize("limit,outcome", [("max_steps", "step_limit"), ("max_requests", "request_limit"),
                                          ("max_tools", "tool_limit")])
def test_persistent_budget_failure_rollback_recovery(tmp_path, limit, outcome):
    db = tmp_path / "rollback.sqlite3"
    normal = ConversationService(db)
    cid = normal.create(A).conversation_id
    normal.respond(cid, "ORD-1001", A)
    restricted = ConversationService(db, policy=ExecutionPolicy(**{limit: 1}))
    with pytest.raises(ExecutionFailure, match=outcome):
        restricted.respond(cid, "where is it?", A)
    assert normal.read(cid, A).turn_count == 1
    response = normal.respond(cid, "where is it?", A)
    assert response.turn_count == 2 and "ORD-1001" in response.response.answer


def test_fake_timeout_persistent_rollback_and_recovery(tmp_path, monkeypatch):
    db = tmp_path / "late.sqlite3"
    normal = ConversationService(db)
    cid = normal.create(A).conversation_id
    normal.respond(cid, "ORD-1001", A)
    clock = FakeClock()
    limited = ConversationService(db, budget_factory=lambda policy: ExecutionBudget(policy, clock=clock))
    original = MockSupportProvider.order_status
    def delayed(self, plan, principal):
        clock.advance(21)
        return original(self, plan, principal)
    with monkeypatch.context() as patch:
        patch.setattr(MockSupportProvider, "order_status", delayed)
        with pytest.raises(ExecutionFailure, match="turn_timeout"):
            limited.respond(cid, "ORD-1002", A)
    assert normal.read(cid, A).turn_count == 1
    assert "ORD-1001" in normal.respond(cid, "where is it?", A).response.answer


def test_real_timeout_persistent_late_result_discarded(tmp_path, monkeypatch):
    db = tmp_path / "real-timeout.sqlite3"
    normal = ConversationService(db)
    cid = normal.create(A).conversation_id
    normal.respond(cid, "ORD-1001", A)
    runner = ReadOnlyRunner(capacity=1)
    limited = ConversationService(db, policy=ExecutionPolicy(tool_seconds=.05),
                                  budget_factory=lambda policy: ExecutionBudget(policy, runner=runner))
    entered, release, exited = Event(), Event(), Event()
    original = MockSupportProvider.order_status
    def blocked(self, plan, principal):
        entered.set()
        try:
            release.wait(5)
            return original(self, plan, principal)
        finally:
            exited.set()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(MockSupportProvider, "order_status", blocked)
            with pytest.raises(ExecutionFailure, match="tool_timeout"):
                limited.respond(cid, "ORD-1002", A)
        assert entered.is_set() and not exited.is_set()
        assert normal.read(cid, A).turn_count == 1
    finally:
        release.set()
        assert exited.wait(1)
    assert runner.slots.acquire(timeout=1)
    runner.slots.release()
    assert normal.read(cid, A).turn_count == 1
    assert "ORD-1001" in normal.respond(cid, "where is it?", A).response.answer


def test_api_persistent_failure_with_foreign_denial(tmp_path, test_event_loop):
    db = tmp_path / "api.sqlite3"
    service = ConversationService(db)
    cid = service.create(A).conversation_id
    service.respond(cid, "ORD-1001", A)
    with TestClient(create_app(Settings(conversation_db=db, execution=ExecutionPolicy(max_steps=1))),
                    backend_options={"loop_factory": lambda: test_event_loop}) as client:
        path = f"/v1/conversations/{cid}/chat"
        failure = client.post(path, json={"message": "where is it?"}, headers={"Authorization": "Bearer demo-customer-a"})
        foreign = client.post(path, json={"message": "where is it?"}, headers={"Authorization": "Bearer demo-customer-b"})
    assert failure.status_code == 503 and failure.json()["detail"]["outcome"] == "step_limit"
    assert foreign.status_code == 404 and foreign.json() == {"detail": "Conversation not found."}
    assert service.read(cid, A).turn_count == 1
