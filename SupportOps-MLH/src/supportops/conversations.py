"""Owner-guarded SQLite conversations and real LangGraph checkpoints.

One connection/saver per operation, never shared between requests. A whole turn
is a single BEGIN IMMEDIATE transaction, including checkpoints and turn count.
The saver's lock protects its graph-worker accesses; no intermediate commits.
"""
from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import tracing_context
from pydantic import BaseModel

from supportops.graph import SupportContext, build_support_graph
from supportops.schemas import ChatRequest, ChatResponse, ConversationResponse, ConversationView, Principal
from supportops.execution import ExecutionBudget, ExecutionPolicy, ExecutionFailure, classify_failure


class ConversationNotFound(Exception):
    """Deliberately identical for foreign and nonexistent conversations."""


class JsonStateSerializer:
    """Data-only JSON: no pickle, dynamic imports, or object reconstruction."""

    def dumps_typed(self, value: Any) -> tuple[str, bytes]:
        def encode(item):
            if isinstance(item, BaseModel):
                return item.model_dump(mode="json")
            if isinstance(item, UUID):
                return str(item)
            raise TypeError(f"Unsupported checkpoint type: {type(item).__name__}")
        return "json", json.dumps(value, default=encode).encode("utf-8")

    def loads_typed(self, value: tuple[str, bytes]) -> Any:
        kind, data = value
        if kind != "json":
            raise ValueError("Unsupported checkpoint encoding")
        return json.loads(data)


def require_owner(conn: sqlite3.Connection, conversation_id: str, principal: Principal) -> int:
    row = conn.execute(
        "SELECT turn_count FROM conversations WHERE id = ? AND owner_id = ?",
        (conversation_id, principal.customer_id),
    ).fetchone()
    if row is None:
        raise ConversationNotFound()
    return row[0]


class OwnedSqliteSaver(SqliteSaver):
    """Request-private saver; every checkpoint cursor revalidates its owner.

Only the service constructs graph configuration. No public arbitrary checkpoint
IDs, namespaces, get_state, or resume(None) interface is exposed.
"""

    def __init__(self, conn: sqlite3.Connection, conversation_id: str, principal: Principal):
        super().__init__(conn, serde=JsonStateSerializer())
        self.conversation_id = conversation_id
        self.principal = principal
        self.is_setup = True  # Schema was initialized before the turn transaction.

    def check_scope(self, config):
        scope = config.get("configurable", {})
        if scope.get("thread_id") != self.conversation_id or scope.get("checkpoint_ns", "") != "":
            raise ConversationNotFound()
        with self.lock:
            require_owner(self.conn, self.conversation_id, self.principal)

    def get_tuple(self, config):
        self.check_scope(config)
        return super().get_tuple(config)

    def put(self, config, checkpoint, metadata, new_versions):
        self.check_scope(config)
        return super().put(config, checkpoint, metadata, new_versions)

    def put_writes(self, config, writes, task_id, task_path=""):
        self.check_scope(config)
        return super().put_writes(config, writes, task_id, task_path)

    def list(self, config, **kwargs):
        # Enumeration/history is deliberately not part of the S02 service.
        raise ConversationNotFound()

    def delete_thread(self, thread_id):
        # No retention/delete operation is authorized by this API yet.
        raise ConversationNotFound()

    @contextmanager
    def cursor(self, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
        with self.lock:
            require_owner(self.conn, self.conversation_id, self.principal)
            with closing(self.conn.cursor()) as cursor:
                yield cursor
        # Deliberately no commit: the service commits only a successful whole turn.


class ConversationService:
    def __init__(self, database: Path, *, policy: ExecutionPolicy | None = None, budget_factory=ExecutionBudget,
                 route_provider=None):
        self.database = Path(database)
        self.policy = policy or ExecutionPolicy()
        self.budget_factory = budget_factory
        self.route_provider = route_provider

    @contextmanager
    def connection(self, budget: ExecutionBudget | None = None) -> Iterator[sqlite3.Connection]:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        # Graph persistence uses worker threads; the saver lock serializes access.
        # This connection is exclusive to one operation and closes after invoke.
        timeout = min(1.0, self.policy.turn_seconds)
        if budget:
            budget.check()
            timeout = min(timeout, self.policy.turn_seconds - (budget.clock() - budget.started))
        with closing(sqlite3.connect(self.database, timeout=timeout, check_same_thread=False)) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Unsupported conversation database schema version")
            conn.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, "
                         "owner_id TEXT NOT NULL, turn_count INTEGER NOT NULL DEFAULT 0, "
                         "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                         "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("CREATE TRIGGER IF NOT EXISTS immutable_conversation_owner "
                         "BEFORE UPDATE OF owner_id, id ON conversations "
                         "BEGIN SELECT RAISE(ABORT, 'Conversation ownership is immutable'); END")
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
            # Schema-only setup, not a checkpoint read or write.
            SqliteSaver(conn, serde=JsonStateSerializer()).setup()
            conn.execute("BEGIN IMMEDIATE")
            try:
                if budget:
                    budget.check()
                yield conn
                if budget:
                    budget.check()
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def create(self, principal: Principal) -> ConversationView:
        conversation_id = uuid4()
        with self.connection() as conn:
            conn.execute("INSERT INTO conversations (id, owner_id) VALUES (?, ?)",
                         (str(conversation_id), principal.customer_id))
        return ConversationView(conversation_id=conversation_id, turn_count=0)

    def read(self, conversation_id: UUID, principal: Principal) -> ConversationView:
        with self.connection() as conn:
            count = require_owner(conn, str(conversation_id), principal)
        # No owner identifiers, private messages, or raw checkpoints in the API.
        return ConversationView(conversation_id=conversation_id, turn_count=count)

    def respond(self, conversation_id: UUID, message: str, principal: Principal) -> ConversationResponse:
        request = ChatRequest(message=message)
        principal = Principal.model_validate(principal)
        budget = self.budget_factory(self.policy)
        try:
            with self.connection(budget) as conn:
                count = require_owner(conn, str(conversation_id), principal)
                saver = OwnedSqliteSaver(conn, str(conversation_id), principal)
                graph = build_support_graph(checkpointer=saver, route_provider=self.route_provider)
                with tracing_context(enabled=False):
                    output = graph.invoke({"message": request.message}, context=SupportContext(principal, budget),
                                          config={"configurable": {"thread_id": str(conversation_id)},
                                                  "recursion_limit": budget.policy.max_steps, "callbacks": []})
                budget.check()
                if not output or output.get("response") is None:
                    raise ExecutionFailure("no_op")
                response = ChatResponse.model_validate(output["response"]).model_copy(update={"request_id": budget.request_id})
                require_owner(conn, str(conversation_id), principal)
                conn.execute("UPDATE conversations SET turn_count = turn_count + 1, "
                             "updated_at = CURRENT_TIMESTAMP WHERE id = ? AND owner_id = ?",
                             (str(conversation_id), principal.customer_id))
            budget.record(response.intent)
            return ConversationResponse(conversation_id=conversation_id, turn_count=count + 1, response=response)
        except Exception as error:
            budget.record("access_denied" if isinstance(error, ConversationNotFound) else classify_failure(error))
            error.supportops_request_id = budget.request_id
            error.supportops_mode = budget.mode
            error.supportops_llm_calls = budget.llm_calls
            if classify_failure(error) == "step_limit" and not isinstance(error, ExecutionFailure):
                bounded = ExecutionFailure("step_limit")
                bounded.supportops_request_id = budget.request_id
                bounded.supportops_mode = budget.mode
                bounded.supportops_llm_calls = budget.llm_calls
                raise bounded from None
            raise
