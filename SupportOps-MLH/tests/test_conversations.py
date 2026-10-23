"""S02 owner, persistence, atomicity, and independent-process acceptance."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import pytest

from supportops.config import Settings
from supportops.conversations import ConversationService, OwnedSqliteSaver, ConversationNotFound, JsonStateSerializer
from supportops.main import create_app
from supportops.schemas import Principal

A = Principal(customer_id="cust_demo_a")
B = Principal(customer_id="cust_demo_b")


@pytest.fixture
def service(tmp_path):
    return ConversationService(tmp_path / "conversations.sqlite3")


@pytest.fixture
def conversation_client(tmp_path, test_event_loop):
    with TestClient(create_app(Settings(conversation_db=tmp_path / "api.sqlite3")),
                    backend_options={"loop_factory": lambda: test_event_loop}) as client:
        yield client


def test_create_server_uuid_and_no_owner_disclosure(conversation_client, auth_a):
    first = conversation_client.post("/v1/conversations", json={}, headers=auth_a)
    second = conversation_client.post("/v1/conversations", json={}, headers=auth_a)
    assert first.status_code == second.status_code == 201
    assert UUID(first.json()["conversation_id"]).version == 4
    assert first.json()["conversation_id"] != second.json()["conversation_id"]
    assert set(first.json()) == {"conversation_id", "turn_count"}
    assert first.json()["turn_count"] == 0


@pytest.mark.parametrize("field", ["customer_id", "owner_id", "conversation_id", "thread_id", "checkpoint_id", "principal"])
def test_create_rejects_forged_fields(conversation_client, auth_a, field):
    assert conversation_client.post("/v1/conversations", json={field: "forged"}, headers=auth_a).status_code == 422


@pytest.mark.parametrize("method,suffix", [("get", ""), ("post", "/chat")])
@pytest.mark.parametrize("owner,foreign", [("auth_a", "auth_b"), ("auth_b", "auth_a")])
def test_foreign_unknown_same_no_checkpoint_access(conversation_client, request, monkeypatch, method, suffix, owner, foreign):
    headers = request.getfixturevalue(owner)
    attacker = request.getfixturevalue(foreign)
    cid = conversation_client.post("/v1/conversations", json={}, headers=headers).json()["conversation_id"]
    def forbidden(*args, **kwargs):
        raise AssertionError("Unauthorized request reached checkpoint saver")
    monkeypatch.setattr(OwnedSqliteSaver, "__init__", forbidden)
    kwargs = {"headers": attacker}
    if method == "post":
        kwargs["json"] = {"message": "Where is it?"}
    results = [getattr(conversation_client, method)(f"/v1/conversations/{value}{suffix}", **kwargs)
               for value in (cid, str(uuid4()))]
    assert [(r.status_code, r.json()) for r in results] == [(404, {"detail": "Conversation not found."})] * 2


@pytest.mark.parametrize("method,suffix", [("get", ""), ("post", "/chat")])
def test_malformed_id_rejected(conversation_client, auth_a, method, suffix):
    kwargs = {"headers": auth_a}
    if method == "post":
        kwargs["json"] = {"message": "where is it"}
    assert getattr(conversation_client, method)(f"/v1/conversations/not-a-uuid{suffix}", **kwargs).status_code == 422


def test_auth_required_and_turn_fields_rejected(conversation_client, auth_a):
    assert conversation_client.post("/v1/conversations", json={}).status_code == 401
    cid = conversation_client.post("/v1/conversations", json={}, headers=auth_a).json()["conversation_id"]
    assert conversation_client.get(f"/v1/conversations/{cid}").status_code == 401
    assert conversation_client.post(f"/v1/conversations/{cid}/chat", json={"message": "Hi"}).status_code == 401
    for field in ("customer_id", "last_order", "plan", "thread_id", "checkpoint_id"):
        assert conversation_client.post(f"/v1/conversations/{cid}/chat", headers=auth_a,
                                        json={"message": "Hi", field: "forged"}).status_code == 422


def test_api_continue_and_metadata(conversation_client, auth_a):
    cid = conversation_client.post("/v1/conversations", json={}, headers=auth_a).json()["conversation_id"]
    for count, message in enumerate(("Where is ORD-1001?", "And when will it arrive?"), 1):
        result = conversation_client.post(f"/v1/conversations/{cid}/chat", headers=auth_a, json={"message": message})
        assert result.status_code == 200
        assert result.json()["turn_count"] == count
        assert "ORD-1001" in result.json()["response"]["answer"]
        assert result.json()["response"]["llm_calls"] == 0
    assert conversation_client.get(f"/v1/conversations/{cid}", headers=auth_a).json()["turn_count"] == 2


@pytest.mark.parametrize("principal,owned,foreign", [(A, "ORD-1001", "ORD-2001"), (B, "ORD-2001", "ORD-1001")])
@pytest.mark.parametrize("attempt", ["Where is {foreign}?", "Refund {foreign}", "Where is ORD-9999?"])
def test_denied_order_cannot_be_followup_memory(service, principal, owned, foreign, attempt):
    cid = service.create(principal).conversation_id
    service.respond(cid, owned, principal)
    denied = service.respond(cid, attempt.format(foreign=foreign), principal).response
    assert denied.intent == "abstain"
    followup = service.respond(cid, "And when will it arrive?", principal).response
    assert followup.intent == "clarify"
    assert foreign not in followup.answer
    assert followup.proposed_action is None


def test_policy_separate_and_same_customer_conversations_isolated(service):
    first, second = [service.create(A).conversation_id for _ in range(2)]
    service.respond(first, "ORD-1001", A)
    policy = service.respond(first, "return policy", A).response
    assert policy.intent == "policy" and policy.citations
    assert service.respond(first, "where is it?", A).response.intent == "order_status"
    assert service.respond(second, "where is it?", A).response.intent == "clarify"
    with sqlite3.connect(service.database) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {"conversations", "checkpoints", "writes"}
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1


def test_followup_rechecks_current_order_owner(service, monkeypatch):
    cid = service.create(A).conversation_id
    service.respond(cid, "ORD-1001", A)
    monkeypatch.setattr("supportops.graph.lookup_order", lambda *args: None)
    assert service.respond(cid, "where is it?", A).response.intent == "clarify"


def test_direct_service_foreign_cannot_modify(service):
    cid = service.create(B).conversation_id
    service.respond(cid, "ORD-2001", B)
    with pytest.raises(ConversationNotFound):
        service.respond(cid, "ORD-1001", A)
    with pytest.raises(ConversationNotFound):
        service.read(cid, A)
    assert service.read(cid, B).turn_count == 1
    assert "ORD-2001" in service.respond(cid, "where is it?", B).response.answer


def test_failed_turn_rolls_back_checkpoints_and_count(service, monkeypatch):
    cid = service.create(A).conversation_id
    service.respond(cid, "ORD-1001", A)
    with sqlite3.connect(service.database) as conn:
        before = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    def fail(*args):
        raise RuntimeError("Synthetic failure")
    monkeypatch.setattr("supportops.provider.MockSupportProvider.order_status", fail)
    with pytest.raises(RuntimeError, match="Synthetic failure"):
        service.respond(cid, "ORD-1002", A)
    with sqlite3.connect(service.database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == before
    assert service.read(cid, A).turn_count == 1


def test_concurrent_turns_serialize_without_lost_updates(service):
    cid = service.create(A).conversation_id
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: service.respond(cid, "ORD-1001", A), range(6)))
    assert sorted(r.turn_count for r in results) == list(range(1, 7))
    assert service.read(cid, A).turn_count == 6


def test_schema_owner_immutable_unknown_version_rejected(service):
    service.create(A)
    with sqlite3.connect(service.database) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE conversations SET owner_id = 'cust_demo_b'")
        conn.execute("PRAGMA user_version = 99")
    with pytest.raises(ValueError, match="schema version"):
        service.create(A)


def test_data_only_serializer_rejects_object_encoding():
    with pytest.raises(ValueError):
        JsonStateSerializer().loads_typed(("pickle", b"malicious"))


def test_saver_guard_checks_owner_and_thread_before_io(service):
    owned = service.create(A).conversation_id
    foreign = service.create(B).conversation_id
    with service.connection() as conn:
        saver = OwnedSqliteSaver(conn, str(owned), A)
        with pytest.raises(ConversationNotFound):
            saver.get_tuple({"configurable": {"thread_id": str(foreign)}})
        saver = OwnedSqliteSaver(conn, str(foreign), A)
        with pytest.raises(ConversationNotFound):
            saver.get_tuple({"configurable": {"thread_id": str(foreign)}})
        with pytest.raises(ConversationNotFound):
            saver.put({"configurable": {"thread_id": str(owned)}}, {}, {}, {})
        with pytest.raises(ConversationNotFound):
            saver.put({"configurable": {"thread_id": str(foreign)}}, {}, {}, {})


def test_checkpoint_memory_is_not_authority_and_identity_not_persisted(service):
    cid = service.create(A).conversation_id
    service.respond(cid, "ORD-1001", A)
    with sqlite3.connect(service.database) as conn:
        rows = conn.execute("SELECT checkpoint_id, checkpoint FROM checkpoints WHERE thread_id=? ORDER BY checkpoint_id DESC",
                            (str(cid),)).fetchall()
        assert rows
        for _, payload in rows:
            assert b"cust_demo_a" not in payload and b"principal" not in payload
        checkpoint = json.loads(rows[0][1])
        checkpoint["channel_values"]["last_order"] = "ORD-2001"
        conn.execute("UPDATE checkpoints SET checkpoint=? WHERE checkpoint_id=? AND thread_id=?",
                     (json.dumps(checkpoint).encode(), rows[0][0], str(cid)))
    result = service.respond(cid, "where is it?", A).response
    assert result.intent == "clarify"
    assert "ORD-2001" not in result.answer


def test_concurrent_customers_keep_independent_memory(service):
    ca, cb = service.create(A).conversation_id, service.create(B).conversation_id
    def turn(args):
        cid, principal, order = args
        service.respond(cid, order, principal)
        return service.respond(cid, "where is it?", principal).response
    with ThreadPoolExecutor(max_workers=2) as pool:
        ra, rb = pool.map(turn, [(ca, A, "ORD-1001"), (cb, B, "ORD-2001")])
    assert "ORD-1001" in ra.answer and "ORD-2001" not in ra.answer
    assert "ORD-2001" in rb.answer and "ORD-1001" not in rb.answer


def test_persistent_tracing_disabled(service, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    def forbidden(*args, **kwargs):
        raise AssertionError("Remote tracing client created")
    monkeypatch.setattr("langsmith.Client.__init__", forbidden)
    cid = service.create(A).conversation_id
    assert service.respond(cid, "ORD-1001", A).response.llm_calls == 0


def test_fresh_process_application_continuity(tmp_path):
    """Two terminated independent interpreters, not just two app instances."""
    script = Path(__file__).parents[1] / "scripts" / "conversation_process_probe.py"
    database = tmp_path / "restart.sqlite3"
    def run(*args):
        result = subprocess.run([sys.executable, str(script), str(database), *args],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    first = run("create")
    second = run("continue", first["conversation_id"])
    assert first["pid"] != second["pid"]
    assert second["response"]["intent"] == "order_status"
    assert "ORD-1001" in second["response"]["answer"]
    assert second["turn_count"] == 2
    assert second["foreign_status"] == 404
