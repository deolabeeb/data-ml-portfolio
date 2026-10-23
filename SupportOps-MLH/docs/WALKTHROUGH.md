# Walk through one request

Aim to understand why each boundary exists, not memorize a script. Begin with `Where is ORD-1001?` from Customer A. Run the six-choice demo before reading further. Line references for this exact candidate are in [CODE_MAP.md](CODE_MAP.md).

## 1 Validate input and establish identity

Open `schemas.py`, then `auth.py`, then `main.py`. `ChatRequest` accepts one bounded, nonblank string. `StrictModel` rejects extra fields, so a client cannot add `customer_id`, `principal` or a graph plan to the request body. The frozen principal stores the customer established by the server's demo-token mapping.

`Depends(require_customer)` supplies that principal to the API handler. A message such as "I am Customer B" changes no authorization. The demonstration tokens are public, so this shows the application boundary, not production identity security.

Explain aloud: "The body says what the customer wants. Authentication establishes who is asking. I do not trust text to choose that identity."

## 2 Plan a fixed route

Open `provider.py`. `MockSupportProvider.plan` extracts an allowed order ID and chooses one of five literal route names. Multiple IDs cause clarification. An order with a refund/cancel keyword creates a review-draft route; otherwise an explicit order goes to status. Policy keywords select one of two allowed sources. Everything else abstains.

`RoutePlan` validates arguments for the selected route. A plan cannot contain arbitrary shell, SQL or a URL. The planner is a deterministic keyword parser; it can misinterpret language. The route plan is a proposal, never permission.

For the example, the plan is `RoutePlan(route="order_status", order_id="ORD-1001")`. Be able to explain why `refund ORD-1001` takes a different branch and why an unrelated order ID may produce a status route followed by an abstention.

## 3 Run an explicit graph

Open `graph.py`. `GraphInput` contains the message; `SupportContext` holds the authenticated principal and request budget. The principal is deliberately not a field in persisted graph state. The compiled graph follows `START -> route -> one operation -> finish -> END`.

`route` validates the plan again at its boundary. `order_status` calls the selected operation with `runtime.context.principal`. `finish` validates the response. Nodes return small state updates instead of modifying authentication. The default stateless graph has no checkpointer.

Explain the distinction between Python type hints and runtime validation: a `TypedDict` helps readers/tools, while Pydantic performs the explicit value checks used here.

## 4 Check ownership inside the tool

Open `tools.py`. The important condition is:

```python
if row["order_id"] == order_id and row["customer_id"] == principal.customer_id:
```

Both comparisons must hold. Returning only an `OrderView` also omits the internal customer ID. If the order is foreign or missing, the tool returns `None` in both cases. The API/provider uses a non-revealing response.

Explain why checking only the order ID is a bug: knowing or guessing an ID would expose another customer's record. Test this idea only in a disposable exercise copy; keep the passing sample unchanged. Read `test_forged_route_plan_does_not_authorize_foreign_order` in `test_graph.py`.

## 5 Add a persistent conversation

Open `conversations.py`. `create` generates a UUID on the server and records its owner. The UUID identifies a conversation; it does not grant access. `require_owner` queries using both conversation ID and principal ID with parameterized SQL. Foreign and nonexistent conversations use the same error.

`OwnedSqliteSaver` guards checkpoint access, validates the server-built thread scope and rechecks ownership. Its cursor does not commit intermediate graph writes. The data-only serializer handles JSON types, UUIDs and Pydantic values; it rejects unsupported encodings instead of loading pickle.

On a follow-up such as "And when will it arrive?", graph memory stores an order reference, not authorization or a cached order result. The reference is looked up under the current principal again. An explicitly denied order clears the previous reference. Only the separate conversation API is stateful; the browser demo and `/v1/chat` stay stateless.

Explain aloud: "Memory helps interpret the next question, but every access must still be authorized."

## 6 Commit only a successful turn

Still in `conversations.py`, follow `connection` and `respond`. The service starts `BEGIN IMMEDIATE`, verifies ownership, runs the graph with the scoped saver, validates the response and increments the turn count. A successful whole turn commits. An exception rolls back checkpoint writes and the turn-count update together.

The saver uses a request-private connection and a lock for its graph-worker accesses. No SQLite connection is passed into the read-only timeout worker. Schema setup is committed before the turn transaction; do not describe failed-turn rollback as rolling back every schema-creation operation.

The demo makes this visible: turn counts advance to 1, then 2; a failed step-limited request does not advance them; recovery succeeds at 3. Read `test_persistent_budget_failure_rollback_recovery` and `test_fresh_process_application_continuity`.

## 7 Bound work and handle errors

Open `execution.py`. `ExecutionPolicy` is immutable server configuration: at most 8 steps, 6 operation requests, 4 tool reads, 2 seconds per operation and 20 seconds per turn. `ExecutionBudget` is a new mutable counter/deadline record for each turn. It checks before dispatch and after results return.

`ReadOnlyRunner` has four retained worker slots. A slow read can time out for the caller while the daemon thread continues. Its slot is released only when the operation exits. This prevents unbounded new workers, but four wedged reads exhaust capacity. A timeout cannot kill arbitrary Python, and cooperative checks do not make SQLite/filesystem I/O universally cancellable.

`classify_failure` converts known failures to fixed categories. `main.py` returns 504 for tool/turn timeout and 503 for other execution failures, with a request UUID and no raw exception text. Unknown/foreign conversations return the same 404. A normal abstention is a successful limited-domain response, not a service crash.

Safe event logs include request ID, outcome and counts, not message contents, customer/order IDs, tokens or raw exceptions. Read `test_logs_correlate_without_untrusted_data` and the real timeout/capacity tests.

## 8 State what this proves

It proves the recorded synthetic behaviours in the tested environment. It does not prove live-model accuracy, production authentication, general prompt-injection resistance, public deployment, financial execution, or your personal mastery. The new mock-only candidate deliberately excludes the separate unfinished live-provider extension; no old test was weakened to repair that issue.

Before an interview, explain one complete successful request, one ownership denial and one rollback test without reading this guide. Then make and explain a small local change yourself. Clearly credit the supplied starter and Codex assistance. The exercises are preparation; they are not a record that you already completed them.
