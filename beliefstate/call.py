from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class LLMCall(BaseModel):
    """Universal representation of an LLM API call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[Dict[str, Any]]
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    system: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Universal representation of an LLM API response."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    raw_response: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
