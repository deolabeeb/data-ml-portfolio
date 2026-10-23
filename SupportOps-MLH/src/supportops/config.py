"""The MLH sample is local and deterministic; live modes fail closed."""
from dataclasses import dataclass, field
from pathlib import Path
import os
from supportops.execution import ExecutionPolicy

@dataclass(frozen=True)
class Settings:
    app_env: str = "local"
    llm_mode: str = "mock"
    conversation_db: Path = Path("data/conversations.sqlite3")
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    def __post_init__(self):
        if self.app_env != "local" or self.llm_mode != "mock":
            raise ValueError("M0 supports local mock mode only; MLH sample has no live adapter")

    @classmethod
    def from_env(cls):
        return cls(app_env=os.environ.get("APP_ENV", "local"),
                   llm_mode=os.environ.get("LLM_MODE", "mock"),
                   conversation_db=Path(os.environ.get("SUPPORTOPS_CONVERSATION_DB", "data/conversations.sqlite3")))
