"""Server-owned bounds for local read-only mock execution, never LLM state.

Timeout bounds caller waiting, NOT cancellation of arbitrary blocking Python.
Daemon worker slots remain occupied until the operation exits: finite exhaustion
fails closed instead of creating unlimited workers. No retries or side effects.
"""
from dataclasses import dataclass
import json
import logging
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from time import monotonic
from typing import Callable, TypeVar
from uuid import uuid4

from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

T = TypeVar("T")
LOGGER = logging.getLogger("supportops.execution")
if not LOGGER.handlers:
    handler = logging.StreamHandler()  # Safe metadata only; stderr, never JSON stdout.
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


class ExecutionFailure(RuntimeError):
    def __init__(self, outcome: str):
        self.outcome = outcome
        super().__init__(outcome)


@dataclass(frozen=True)
class ExecutionPolicy:
    max_steps: int = 8
    max_requests: int = 6
    max_tools: int = 4
    tool_seconds: float = 2.0
    turn_seconds: float = 20.0

    def __post_init__(self):
        if any(type(value) is not int or not 1 <= value <= 32 for value in
               (self.max_steps, self.max_requests, self.max_tools)):
            raise ValueError("Execution counters must be integers from 1 to 32")
        if any(not 0 < value <= 20 for value in (self.tool_seconds, self.turn_seconds)):
            raise ValueError("Execution time limits must be positive and at most 20 seconds")


class ReadOnlyRunner:
    """Bounded daemon workers; never pass SQLite connections or mutable state."""
    def __init__(self, capacity: int = 4):
        self.slots = BoundedSemaphore(capacity)

    def run(self, operation: Callable[[], T], timeout: float) -> T:
        if not self.slots.acquire(blocking=False):
            raise ExecutionFailure("capacity_exhausted")
        result: Queue = Queue(maxsize=1)

        def work():
            try:
                try:
                    completed = (True, operation())
                except Exception as error:
                    completed = (False, error)
            finally:
                self.slots.release()
            result.put(completed)

        worker = Thread(target=work, name="supportops-read-only", daemon=True)
        try:
            worker.start()
        except BaseException:
            self.slots.release()
            raise
        try:
            success, value = result.get(timeout=timeout)
        except Empty:
            raise ExecutionFailure("tool_timeout") from None
        if not success:
            raise value
        return value


DEFAULT_RUNNER = ReadOnlyRunner()  # Global finite cap, including timed-out workers.


def classify_failure(error: Exception) -> str:
    if isinstance(error, ExecutionFailure):
        return error.outcome
    if isinstance(error, GraphRecursionError):
        return "step_limit"
    if isinstance(error, ValidationError):
        return "malformed_output"
    if isinstance(error, TimeoutError):
        return "tool_timeout"
    return "tool_failure"


class ExecutionBudget:
    def __init__(self, policy: ExecutionPolicy | None = None, *, clock=monotonic,
                 runner: ReadOnlyRunner | None = None):
        self.policy = policy or ExecutionPolicy()
        self.clock = clock
        self.runner = runner or DEFAULT_RUNNER
        self.started = clock()
        self.request_id = uuid4()
        self.steps = self.requests = self.tools = 0
        self.last_operation = "none"
        self.mode = "mock"
        self.llm_calls = 0

    def check(self):
        if self.clock() - self.started >= self.policy.turn_seconds:
            raise ExecutionFailure("turn_timeout")

    def step(self):
        self.check()
        self.steps += 1
        if self.steps > self.policy.max_steps:
            raise ExecutionFailure("step_limit")

    def call(self, name: str, operation: Callable[[], T], *, tool: bool = True) -> T:
        self.check()
        self.last_operation = name  # Only fixed server-defined operation names.
        self.requests += 1
        self.tools += int(tool)
        if self.requests > self.policy.max_requests:
            raise ExecutionFailure("request_limit")
        if self.tools > self.policy.max_tools:
            raise ExecutionFailure("tool_limit")
        remaining = self.policy.turn_seconds - (self.clock() - self.started)
        allowance = min(self.policy.tool_seconds, remaining)
        before = self.clock()
        try:
            value = self.runner.run(operation, allowance)
        except ExecutionFailure as error:
            if error.outcome == "tool_timeout" and remaining <= self.policy.tool_seconds:
                raise ExecutionFailure("turn_timeout") from None
            raise
        self.check()
        if self.clock() - before >= self.policy.tool_seconds:
            raise ExecutionFailure("tool_timeout")
        if value is None and name not in {"order_lookup", "memory_lookup"}:
            raise ExecutionFailure("no_op")
        return value

    def record(self, outcome: str):
        # No message, owner, conversation/order ID, exception, token or reasoning.
        LOGGER.info(json.dumps({"request_id": str(self.request_id), "mode": self.mode,
                                "outcome": outcome, "operation": self.last_operation,
                                "steps": self.steps, "requests": self.requests, "tools": self.tools,
                                "elapsed_ms": round(max(0, self.clock() - self.started) * 1000, 3),
                                "llm_calls": self.llm_calls, "retries": 0}))
