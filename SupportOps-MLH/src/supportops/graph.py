"""Typed routing: stateless by default, owner-guarded checkpointing in S02.

Only the message enters the public graph input. Authorization comes from the
server-owned, immutable runtime context; no customer identity is a state field.
"""
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langsmith import tracing_context

from supportops.provider import MockSupportProvider, Route, RoutePlan, RouteProvider
from supportops.schemas import ChatRequest, ChatResponse, Principal
from supportops.tools import lookup_order
from supportops.execution import ExecutionBudget, ExecutionPolicy, ExecutionFailure, classify_failure


class GraphInput(TypedDict):
    message: str


class GraphOutput(TypedDict):
    response: ChatResponse


class GraphState(GraphInput, total=False):
    plan: RoutePlan
    response: ChatResponse
    last_order: str | None


class GraphUpdate(TypedDict, total=False):
    plan: RoutePlan
    response: ChatResponse
    last_order: str | None


@dataclass(frozen=True)
class SupportContext:
    principal: Principal
    budget: ExecutionBudget | None = None


def build_support_graph(provider: MockSupportProvider | None = None, *,
                        route_provider: RouteProvider | None = None,
                        checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    operations = provider if provider is not None else MockSupportProvider()

    def call(runtime, name, operation, *, tool=True):
        budget = runtime.context.budget
        return budget.call(name, operation, tool=tool) if budget else operation()

    def step(runtime):
        if runtime.context.budget:
            runtime.context.budget.step()

    def route(state: GraphState, runtime: Runtime[SupportContext]) -> GraphUpdate:
        step(runtime)
        # Validate at this boundary too: direct graph callers cannot bypass the
        # API's message contract. Only the server's route planner chooses a node.
        message = ChatRequest(message=state["message"]).message
        if checkpointer is not None and message.lower().rstrip("?.!") in {
            "and when will it arrive", "when will it arrive", "where is it",
            "what is its status", "refund it", "cancel it",
        }:
            last_order = state.get("last_order")
            # Memory is only a reference, never permission or a cached order result.
            if not last_order or call(runtime, "memory_lookup", lambda: lookup_order(runtime.context.principal, last_order)) is None:
                return {"plan": RoutePlan(route="clarify", reason="missing_order_id"), "last_order": None}
            message = f"{message} {last_order}"
        if route_provider is not None:
            if runtime.context.budget is None:
                raise ExecutionFailure("missing_budget")
            proposed = route_provider.plan(message, runtime.context.budget)
        else:
            proposed = call(runtime, "route_plan", lambda: operations.plan(message), tool=False)
        # Revalidate even model instances (including model_construct/copy bypasses).
        plan = RoutePlan.model_validate(proposed.model_dump() if isinstance(proposed, RoutePlan) else proposed)
        return {"plan": plan}

    def order_status(state: GraphState, runtime: Runtime[SupportContext]) -> GraphUpdate:
        step(runtime)
        return {"response": call(runtime, "order_status", lambda: operations.order_status(state["plan"], runtime.context.principal))}

    def review_request(state: GraphState, runtime: Runtime[SupportContext]) -> GraphUpdate:
        step(runtime)
        return {"response": call(runtime, "review_request", lambda: operations.review_request(state["plan"], runtime.context.principal))}

    def policy(state: GraphState, runtime: Runtime[SupportContext]) -> GraphUpdate:
        step(runtime)
        return {"response": call(runtime, "policy", lambda: operations.policy(state["plan"]))}

    def clarify(state: GraphState, runtime: Runtime[SupportContext]) -> GraphUpdate:
        step(runtime)
        response = call(runtime, "clarify", lambda: operations.clarify(state["plan"]), tool=False)
        if checkpointer is not None and state["plan"].reason == "missing_order_id":
            response = response.model_copy(update={"answer": "Please provide an order ID from your account."})
        return {"response": response}

    def abstain(state: GraphState, runtime: Runtime[SupportContext]) -> GraphUpdate:
        step(runtime)
        return {"response": call(runtime, "abstain", lambda: operations.abstain(state["plan"]), tool=False)}

    def finish(state: GraphState, runtime: Runtime[SupportContext]) -> GraphUpdate:
        step(runtime)
        response = ChatResponse.model_validate(state["response"])
        if runtime.context.budget:
            response = response.model_copy(update={"mode": runtime.context.budget.mode,
                                                   "llm_calls": runtime.context.budget.llm_calls})
        update: GraphUpdate = {"response": response}
        if checkpointer is not None:
            plan = RoutePlan.model_validate(state["plan"])
            if plan.order_id:
                # A denied explicit reference clears prior context conservatively.
                update["last_order"] = (plan.order_id if response.intent in {
                    "order_status", "review_request"
                } and call(runtime, "order_lookup", lambda: lookup_order(runtime.context.principal, plan.order_id)) is not None else None)
        return update

    def next_route(state: GraphState) -> Route:
        return state["plan"].route

    builder = StateGraph(GraphState, input_schema=GraphInput,
                         output_schema=GraphOutput, context_schema=SupportContext)
    builder.add_node("route", route)
    for name, node in {"order_status": order_status, "review_request": review_request,
                       "policy": policy, "clarify": clarify, "abstain": abstain}.items():
        builder.add_node(name, node)
        builder.add_edge(name, "finish")
    builder.add_node("finish", finish)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", next_route,
                                  {name: name for name in (
                                      "order_status", "review_request", "policy", "clarify", "abstain")})
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)  # Stateless remains the default.


class GraphSupportProvider:
    def __init__(self, provider: MockSupportProvider | None = None, *, policy: ExecutionPolicy | None = None,
                 budget_factory=ExecutionBudget, route_provider: RouteProvider | None = None) -> None:
        self.graph = build_support_graph(provider, route_provider=route_provider)
        self.policy = policy or ExecutionPolicy()
        self.budget_factory = budget_factory

    def respond(self, message: str, principal: Principal) -> ChatResponse:
        request = ChatRequest(message=message)
        budget = self.budget_factory(self.policy)
        context = SupportContext(principal=Principal.model_validate(principal), budget=budget)
        # Disable remote tracing even if inherited environment opts into it.
        # No user-supplied callbacks, context, config or state reach this call.
        try:
            with tracing_context(enabled=False):
                output = self.graph.invoke({"message": request.message},
                                           config={"recursion_limit": budget.policy.max_steps, "callbacks": []},
                                           context=context)
            budget.check()
            if not output or output.get("response") is None:
                raise ExecutionFailure("no_op")
            response = ChatResponse.model_validate(output["response"]).model_copy(update={"request_id": budget.request_id})
            budget.record(response.intent)
            return response
        except Exception as error:
            budget.record(classify_failure(error))
            error.supportops_request_id = budget.request_id
            error.supportops_mode = budget.mode
            error.supportops_llm_calls = budget.llm_calls
            if classify_failure(error) == "step_limit" and not isinstance(error, ExecutionFailure):
                bounded = ExecutionFailure("step_limit")
                bounded.supportops_request_id = budget.request_id
                bounded.supportops_mode = budget.mode
                bounded.supportops_llm_calls = budget.llm_calls
                raise bounded from None
            # Internal callers keep typed/original errors; the API renders safe detail.
            raise
