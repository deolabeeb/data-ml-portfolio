"""The distribution must stay mock-only and load this candidate's source."""
import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError
import supportops
from supportops.config import Settings
from supportops.schemas import ChatResponse


@pytest.mark.parametrize('kwargs', [{'llm_mode': 'openai'}, {'app_env': 'production'}])
def test_live_or_public_mode_rejected(kwargs):
    with pytest.raises(ValueError, match='local mock mode only'):
        Settings(**kwargs)


def test_response_cannot_claim_live_model_execution():
    with pytest.raises(ValidationError):
        ChatResponse(request_id=uuid4(), intent='abstain', answer='No answer',
                     mode='openai', llm_calls=1)


def test_distribution_has_no_live_adapter():
    assert importlib.util.find_spec('supportops.openai_provider') is None
    assert importlib.util.find_spec('supportops.provider_authorization') is None


def test_candidate_source_is_used():
    assert Path(supportops.__file__).resolve().parent == (
        Path(__file__).resolve().parents[1] / 'src/supportops')
    assert supportops.__version__ == '0.1.0+mlh.mock1'
