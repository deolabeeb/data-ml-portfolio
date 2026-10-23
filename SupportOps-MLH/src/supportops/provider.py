"""Deterministic route planning and response operations; no model calls.

Purpose: establish input, tool, and response contracts without API spending.
The message rules below are intentionally limited and must never be advertised
as natural-language understanding or as a prompt-injection defense.
"""
import re
from pydantic import Field, model_validator
from typing import Literal, Protocol
from uuid import uuid4
from supportops.schemas import ChatResponse, Principal, ProposedAction, StrictModel
from supportops.tools import lookup_order, read_policy
from supportops.execution import ExecutionBudget


class SupportProvider(Protocol):
    def respond(self, message: str, principal: Principal) -> ChatResponse: ...


class RouteProvider(Protocol):
    """An injected planner receives text and a server budget, never authority.

    Implementations must return a RoutePlan; the graph revalidates it and invokes
    only fixed server operations. Live implementations must cancel their own I/O.
    """
    def plan(self, message: str, budget: ExecutionBudget) -> "RoutePlan": ...


Route = Literal["order_status", "policy", "review_request", "clarify", "abstain"]


class RoutePlan(StrictModel):
    route: Route
    order_id: str | None = Field(default=None, pattern=r"^ORD-\d{4}$")
    source_id: Literal["returns-v1", "shipping-v1"] | None = None
    draft_kind: Literal["refund_review", "cancellation_review"] | None = None
    reason: Literal["multiple_orders", "missing_order_id"] | None = None

    @model_validator(mode="after")
    def required_route_fields(self) -> "RoutePlan":
        if self.route in {"order_status", "review_request"} and self.order_id is None:
            raise ValueError("An order route requires an order ID")
        if self.route == "review_request" and self.draft_kind is None:
            raise ValueError("A draft route requires its kind")
        if self.route == "policy" and self.source_id is None:
            raise ValueError("A policy route requires an allowed source ID")
        if self.route == "clarify" and self.reason is None:
            raise ValueError("A clarification requires its reason")
        allowed = {"order_status": {"order_id"}, "review_request": {"order_id", "draft_kind"},
                   "policy": {"source_id"}, "clarify": {"reason"}, "abstain": set()}[self.route]
        if any(getattr(self, key) is not None and key not in allowed
               for key in ("order_id", "source_id", "draft_kind", "reason")):
            raise ValueError("Arguments must belong to the selected fixed route")
        return self


class MockRouteProvider:
    def plan(self, message: str, budget: ExecutionBudget) -> RoutePlan:
        return budget.call("route_plan", lambda: MockSupportProvider().plan(message), tool=False)


class MockSupportProvider:
    def plan(self, message: str) -> RoutePlan:
        """Preserve M0 keyword precedence; this plan grants no authorization."""
        text = message.lower()
        words = set(re.findall(r"[a-z]+", text))
        matches = list(dict.fromkeys(re.findall(r"\bORD-\d{4}\b", message.upper())))
        if len(matches) > 1:
            return RoutePlan(route="clarify", reason="multiple_orders")
        if matches:
            if words & {"refund", "cancel"}:
                kind = "refund_review" if "refund" in words else "cancellation_review"
                return RoutePlan(route="review_request", order_id=matches[0], draft_kind=kind)
            return RoutePlan(route="order_status", order_id=matches[0])
        if words & {"order", "tracking"}:
            return RoutePlan(route="clarify", reason="missing_order_id")
        source_id = None
        if words & {"return", "returns", "refund", "refunds"}:
            source_id = "returns-v1"
        elif words & {"shipping", "delivery", "dispatch"}:
            # Only the generic fixture policy is returned; this makes no new promise.
            source_id = "shipping-v1"
        if source_id:
            return RoutePlan(route="policy", source_id=source_id)
        return RoutePlan(route="abstain")

    def order_status(self, plan: RoutePlan, principal: Principal) -> ChatResponse:
        assert plan.order_id is not None  # Validated by RoutePlan.
        order = lookup_order(principal, plan.order_id)
        if order is None:
            return self.missing_order()
        return ChatResponse(request_id=uuid4(), intent="order_status",
                            answer=(f"Synthetic order {order.order_id}: {order.status}. "
                                    f"Delivery: {order.estimated_delivery}."))

    def review_request(self, plan: RoutePlan, principal: Principal) -> ChatResponse:
        # Even a forged plan must pass ownership inside the read-only tool.
        assert plan.order_id is not None and plan.draft_kind is not None
        order = lookup_order(principal, plan.order_id)
        if order is None:
            return self.missing_order()
        return ChatResponse(
            request_id=uuid4(), intent="review_request",
            answer=("This is a review-request draft only. It has NOT been saved, "
                    "submitted, approved, or executed. No refund or cancellation occurred."),
            proposed_action=ProposedAction(kind=plan.draft_kind, order_id=order.order_id),
        )

    def policy(self, plan: RoutePlan) -> ChatResponse:
        assert plan.source_id is not None
        citation = read_policy(plan.source_id)
        return ChatResponse(request_id=uuid4(), intent="policy",
                            answer="The fictional policy says: " + citation.quote,
                            citations=[citation])

    def clarify(self, plan: RoutePlan) -> ChatResponse:
        answer = ("Please ask about one order at a time in this local demo."
                  if plan.reason == "multiple_orders" else
                  "Please provide an order ID such as ORD-1001. Access is checked against your demo account.")
        return ChatResponse(request_id=uuid4(), intent="clarify", answer=answer)

    def abstain(self, plan: RoutePlan) -> ChatResponse:
        return ChatResponse(
            request_id=uuid4(),
            intent="abstain",
            answer=("This limited mock demo has no verified answer to that question. "
                    "Human escalation will be implemented in a later milestone."),
        )

    @staticmethod
    def missing_order() -> ChatResponse:
        return ChatResponse(request_id=uuid4(), intent="abstain",
                            answer="I could not find that order in your account.")

    def respond(self, message: str, principal: Principal) -> ChatResponse:
        """Retain the provider seam for injected/mock callers; API uses the graph."""
        plan = self.plan(message)
        if plan.route == "order_status":
            return self.order_status(plan, principal)
        if plan.route == "review_request":
            return self.review_request(plan, principal)
        if plan.route == "policy":
            return self.policy(plan)
        if plan.route == "clarify":
            return self.clarify(plan)
        return self.abstain(plan)
