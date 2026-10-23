"""Local-only FastAPI entry point; no provider clients or external side effects."""
from importlib.resources import files
from typing import Annotated
from uuid import UUID, uuid4
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from supportops.auth import require_customer
from supportops.config import Settings
from supportops.conversations import ConversationService, ConversationNotFound
from supportops.graph import GraphSupportProvider
from supportops.provider import SupportProvider, RouteProvider, MockRouteProvider
from supportops.schemas import ChatRequest, ChatResponse, OrderView, Principal
from supportops.schemas import ConversationCreate, ConversationView, ConversationResponse
from supportops.execution import ExecutionBudget, classify_failure


def create_app(
    settings: Settings | None = None,
    provider: SupportProvider | None = None,
    *, route_provider: RouteProvider | None = None,
) -> FastAPI:
    config = settings or Settings.from_env()
    if route_provider is None:
        route_provider = MockRouteProvider()
    active_provider = provider if provider is not None else GraphSupportProvider(
        policy=config.execution, route_provider=route_provider)
    conversations = ConversationService(config.conversation_db, policy=config.execution, route_provider=route_provider)

    def failure(error: Exception) -> JSONResponse:
        outcome = classify_failure(error)
        return JSONResponse(status_code=504 if outcome in {"tool_timeout", "turn_timeout", "provider_timeout"} else 503,
                            content={"detail": {"outcome": outcome,
                                               "request_id": str(getattr(error, "supportops_request_id", uuid4())),
                                               "mode": getattr(error, "supportops_mode", "mock"),
                                               "llm_calls": getattr(error, "supportops_llm_calls", 0),
                                               "action_executed": False}})
    api = FastAPI(
        title="SupportOps Agent — local mock foundation",
        version="0.1.0",
        description="Synthetic mock with owner-scoped SQLite conversations. No live AI or action execution.",
    )
    api.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )

    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo_page() -> str:
        return files("supportops").joinpath("static/index.html").read_text(encoding="utf-8")

    @api.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": config.llm_mode, "milestone": "M0", "data": "synthetic"}

    @api.get("/v1/orders/{order_id}", response_model=OrderView)
    def order_lookup(
        order_id: str,
        principal: Annotated[Principal, Depends(require_customer)],
    ) -> OrderView:
        from supportops.tools import lookup_order
        budget = ExecutionBudget(config.execution)
        try:
            order = budget.call("order_lookup", lambda: lookup_order(principal, order_id.upper()))
            budget.record("order_status" if order else "abstain")
        except Exception as error:
            budget.record(classify_failure(error))
            error.supportops_request_id = budget.request_id
            return failure(error)
        if order is None:
            raise HTTPException(status_code=404, detail="Order not found in your account.")
        return order

    @api.post("/v1/chat", response_model=ChatResponse)
    def chat(
        body: ChatRequest,
        principal: Annotated[Principal, Depends(require_customer)],
    ) -> ChatResponse:
        try:
            return active_provider.respond(body.message, principal)
        except Exception as error:
            return failure(error)

    @api.post("/v1/conversations", response_model=ConversationView, status_code=201)
    def create_conversation(body: ConversationCreate,
                            principal: Annotated[Principal, Depends(require_customer)]):
        try:
            return conversations.create(principal)
        except Exception as error:
            return failure(error)

    @api.get("/v1/conversations/{conversation_id}", response_model=ConversationView)
    def read_conversation(conversation_id: UUID,
                          principal: Annotated[Principal, Depends(require_customer)]):
        try:
            return conversations.read(conversation_id, principal)
        except ConversationNotFound:
            raise HTTPException(404, "Conversation not found.") from None
        except Exception as error:
            return failure(error)

    @api.post("/v1/conversations/{conversation_id}/chat", response_model=ConversationResponse)
    def conversation_chat(conversation_id: UUID, body: ChatRequest,
                          principal: Annotated[Principal, Depends(require_customer)]):
        try:
            return conversations.respond(conversation_id, body.message, principal)
        except ConversationNotFound:
            raise HTTPException(404, "Conversation not found.") from None
        except Exception as error:
            return failure(error)

    return api


app = create_app()
