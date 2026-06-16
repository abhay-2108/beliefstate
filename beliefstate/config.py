from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict

DEFAULT_EXTRACT_PROMPT = """You are an expert belief state extractor. From this text, extract all factual claims being asserted.

CRITICAL RULES FOR SUBJECT NORMALIZATION:
1. First-person pronouns (I, me, my, mine, we) MUST be mapped to "USER".
2. Second-person pronouns (you, your) referring to the AI MUST be mapped to "ASSISTANT".
3. Never use the user's name as the subject for their own traits; always map it to "USER".

Return ONLY a JSON array of objects.

Format:
[
  {{"subject": "...", "predicate": "...", "value": "...", "confidence": 0.0-1.0, "source": "..."}}
]

Example Input: "My name is Bob and I live in Paris."
Example Output: 
[
  {{"subject": "User", "predicate": "name is", "value": "Bob", "confidence": 1.0, "source": "user"}},
  {{"subject": "User", "predicate": "lives in", "value": "Paris", "confidence": 1.0, "source": "user"}}
]

If no claims, return [].

Text:
{response}
"""

DEFAULT_JUDGE_PROMPT = """Do these two claims contradict or entail each other?
Premise: {premise}
Hypothesis: {hypothesis}

Analyze the relationship between them.
Return ONLY a JSON object with keys 'relationship' ('contradiction', 'entailment', or 'neutral'), 'score' (float between 0.0 and 1.0 representing confidence), and 'reason' (string explanation).
Do not wrap in markdown.

Format:
{{"relationship": "contradiction", "score": 0.9, "reason": "..."}}
"""


class TrackerConfig(BaseModel):
    """Configuration for the BeliefTracker."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Store settings
    store_type: str = Field(
        default="sqlite", description="Type of storage to use ('sqlite', 'redis')."
    )
    store_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the store."
    )

    # Detection settings
    similarity_threshold: float = Field(
        default=0.82, description="Threshold for embedding similarity."
    )
    contradiction_threshold: float = Field(
        default=0.70, description="Threshold for finding contradictions."
    )

    # Prompts
    extract_prompt_template: str = Field(
        default=DEFAULT_EXTRACT_PROMPT, description="Prompt used to extract beliefs."
    )
    judge_prompt_template: str = Field(
        default=DEFAULT_JUDGE_PROMPT,
        description="Prompt used to detect contradictions.",
    )

    # Task behavior
    enable_background_tasks: bool = Field(
        default=True, description="Run tracking async to avoid blocking."
    )

    # Internal override for the tracker
    # By default, tracking uses the same provider as the user's wrapped LLM call.
    # Users can set this to use a specific LLM provider (e.g. OpenAIAdapter) for extraction/detection.
    internal_provider: Optional[Any] = Field(
        default=None, description="Explicit provider for tracker's internal LLM calls."
    )

    # Resilience settings
    retry_max_attempts: int = Field(
        default=5, description="Max retry attempts for LLM API calls."
    )
    retry_min_wait: float = Field(
        default=2.0, description="Minimum wait time between retries in seconds."
    )
    retry_max_wait: float = Field(
        default=30.0, description="Maximum wait time between retries in seconds."
    )
    retry_multiplier: float = Field(
        default=2.0, description="Multiplier for exponential backoff."
    )

    enable_circuit_breaker: bool = Field(
        default=True, description="Enable circuit breaker protection."
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5, description="Number of failures before tripping circuit breaker."
    )
    circuit_breaker_recovery_timeout: float = Field(
        default=30.0, description="Cooldown time in seconds before attempting recovery."
    )

    # Pluggable dispatcher settings
    task_dispatcher_type: str = Field(
        default="asyncio",
        description="Task dispatcher type ('asyncio', 'sync', 'celery', 'rq').",
    )
    dispatcher_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments/instances for initializing dispatcher.",
    )
