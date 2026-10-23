"""S01 acceptance: actual node execution, immutable identity and M0 contracts."""
from dataclasses import FrozenInstanceError
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph.state import CompiledStateGraph
import langsmith
from pydantic import ValidationError
import pytest

from supportops.graph import GraphSupportProvider, SupportContext, build_support_graph
from supportops.provider import MockSupportProvider, RoutePlan
from supportops.schemas import ChatResponse, Principal

PRINCIPAL_A = Principal(customer_id="cust_demo_a")
PRINCIPAL_B = Principal(customer_id="cust_demo_b")


@pytest.mark.parametrize("message,node,intent", [
    ("Where is ORD-1001?", "order_status", "order_status"),
    ("where is ord-1001?", "order_status", "order_status"),
    ("What is your return policy?", "policy", "policy"),
    ("What is the shipping policy?", "policy", "policy"),
    ("Please refund ORD-1001", "review_request", "review_request"),
    ("Cancel ORD-1001", "review_request", "review_request"),
    ("Where is my order?", "clarify", "clarify"),
    ("Compare ORD-1001 and ORD-2001", "clarify", "clarify"),
    ("Can you repair my laptop?", "abstain", "abstain"),
    ("Where is ORD-2001?", "order_status", "abstain"),
    ("Refund ORD-2001", "review_request", "abstain"),
])
def test_actual_graph_executes_only_selected_route(message, node, intent):
    graph = build_support_graph()
    assert isinstance(graph, CompiledStateGraph)
    with langsmith.tracing_context(enabled=False):
        updates = list(graph.stream({"message": message}, context=SupportContext(PRINCIPAL_A),
                                    stream_mode="updates"))
    assert [name for update in updates for name in update] == ["route", node, "finish"]
    response = updates[-1]["finish"]["response"]
    assert isinstance(response, ChatResponse)
    assert response.intent == intent
    assert response.llm_calls == 0 and response.action_executed is False
    assert graph.checkpointer is None


@pytest.mark.parametrize("customer_id", ["cust_demo_a", "cust_demo_b"])
def test_graph_input_cannot_supply_principal_or_plan(customer_id):
    graph = build_support_graph()
    with langsmith.tracing_context(enabled=False):
        result = graph.invoke({
            "message": "Where is ORD-2001?",
            "customer_id": customer_id,
            "principal": PRINCIPAL_B,
            "plan": {"route": "policy", "source_id": "returns-v1"},
            "response": {"intent": "order_status", "answer": "delivered"},
        }, context=SupportContext(PRINCIPAL_A))
    assert set(result) == {"response"}
    assert result["response"].intent == "abstain"
    assert "delivered" not in result["response"].answer


def test_runtime_context_and_principal_are_immutable():
    context = SupportContext(PRINCIPAL_A)
    with pytest.raises(FrozenInstanceError):
        context.principal = PRINCIPAL_B
    with pytest.raises(ValidationError):
        context.principal.customer_id = "cust_demo_b"


@pytest.mark.parametrize("route,extra", [
    ("order_status", {}),
    ("review_request", {"draft_kind": "refund_review"}),
])
def test_forged_route_plan_does_not_authorize_foreign_order(route, extra):
    class ForgedPlanner(MockSupportProvider):
        def plan(self, message):
            return RoutePlan(route=route, order_id="ORD-2001", **extra)

    response = GraphSupportProvider(ForgedPlanner()).respond("Hello", PRINCIPAL_A)
    assert response.intent == "abstain"
    assert response.proposed_action is None
    assert "delivered" not in response.answer


def test_planner_cannot_add_an_identity_field():
    with pytest.raises(ValidationError):
        RoutePlan(route="abstain", customer_id="cust_demo_b")


@pytest.mark.parametrize("body", [
    {"customer_id": "cust_demo_b"},
    {"principal": {"customer_id": "cust_demo_b"}},
    {"context": {"principal": {"customer_id": "cust_demo_b"}}},
    {"plan": {"route": "order_status", "order_id": "ORD-2001"}},
])
def test_api_rejects_graph_context_fields(client, auth_a, body):
    response = client.post("/v1/chat", headers=auth_a,
                           json={"message": "Where is ORD-2001?", **body})
    assert response.status_code == 422


def test_shared_graph_does_not_retain_previous_customer():
    provider = GraphSupportProvider()
    assert provider.respond("Where is ORD-2001?", PRINCIPAL_B).intent == "order_status"
    assert provider.respond("Where is ORD-2001?", PRINCIPAL_A).intent == "abstain"
    assert provider.respond("Where is ORD-1001?", PRINCIPAL_A).intent == "order_status"
    assert provider.respond("Where is ORD-1001?", PRINCIPAL_B).intent == "abstain"


@pytest.mark.parametrize("message", ["Where is ORD-1001?", "Return policy?", "Refund ORD-1001",
                                    "Where is my order?", "Repair my laptop"])
def test_response_is_deterministic_except_request_id(message):
    provider = GraphSupportProvider()
    first = provider.respond(message, PRINCIPAL_A)
    second = provider.respond(message, PRINCIPAL_A)
    assert first.request_id != second.request_id
    assert first.model_dump(exclude={"request_id"}) == second.model_dump(exclude={"request_id"})


def test_concurrent_graph_invocations_keep_separate_principals():
    provider = GraphSupportProvider()
    principals = [PRINCIPAL_A, PRINCIPAL_B] * 8
    with ThreadPoolExecutor(max_workers=4) as workers:
        responses = list(workers.map(
            lambda principal: provider.respond("Where is ORD-2001?", principal), principals))
    assert [response.intent for response in responses] == ["abstain", "order_status"] * 8


def test_api_default_calls_compiled_graph(client, auth_a, monkeypatch):
    original = CompiledStateGraph.invoke
    calls = []

    def invoke(self, *args, **kwargs):
        calls.append(kwargs["context"].principal)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CompiledStateGraph, "invoke", invoke)
    assert client.post("/v1/chat", headers=auth_a,
                       json={"message": "Where is ORD-1001?"}).status_code == 200
    assert calls == [PRINCIPAL_A]


def test_graph_operations_do_not_call_legacy_dispatcher():
    class OperationsOnly(MockSupportProvider):
        def respond(self, message, principal):
            pytest.fail("The graph must execute its own selected operation node")

    assert GraphSupportProvider(OperationsOnly()).respond("Where is ORD-1001?", PRINCIPAL_A).intent == "order_status"


def test_inherited_remote_tracing_is_disabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    def forbidden_client(*args, **kwargs):
        pytest.fail("Offline mock must not construct a remote tracing client")

    monkeypatch.setattr(langsmith.Client, "__init__", forbidden_client)
    assert GraphSupportProvider().respond("Hello", PRINCIPAL_A).intent == "abstain"


@pytest.mark.parametrize("plan", [
    {"route": "shell"},
    {"route": "order_status"},
    {"route": "review_request", "order_id": "ORD-1001"},
    {"route": "policy", "source_id": "https://example.com"},
])
def test_route_plan_validates_allowed_operations(plan):
    with pytest.raises(ValidationError):
        RoutePlan.model_validate(plan)
