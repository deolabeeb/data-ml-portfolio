"""Public contracts; authorization context is never accepted in the request body."""
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Principal(StrictModel):
    customer_id: str


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=2000, strict=True)

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Message cannot be blank.")
        return clean


class Citation(StrictModel):
    source_id: str
    title: str
    quote: str


class OrderView(StrictModel):
    order_id: str
    status: str
    estimated_delivery: str
    synthetic: Literal[True] = True


class ProposedAction(StrictModel):
    kind: Literal["refund_review", "cancellation_review"]
    order_id: str
    status: Literal["draft_only_not_saved"] = "draft_only_not_saved"


class ChatResponse(StrictModel):
    request_id: UUID
    mode: Literal["mock"] = "mock"
    intent: Literal["order_status", "policy", "review_request", "clarify", "abstain"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    proposed_action: ProposedAction | None = None
    action_executed: Literal[False] = False
    llm_calls: Literal[0] = 0


class ConversationCreate(StrictModel):
    """No client-supplied identity, UUID, or graph configuration."""


class ConversationView(StrictModel):
    conversation_id: UUID
    turn_count: int


class ConversationResponse(ConversationView):
    response: ChatResponse
