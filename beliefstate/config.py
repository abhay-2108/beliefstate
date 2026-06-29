from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

DEFAULT_EXTRACT_PROMPT = """
You are a precise fact extraction engine. Extract ONLY facts that the USER
explicitly stated. Do NOT extract facts the assistant mentioned, suggested,
or provided as general knowledge.

QUALITY OVER QUANTITY — target 1-3 beliefs per turn. Most turns yield 0-1
beliefs. Only extract when the user shares a concrete, new piece of information.

RULES — ONLY EXTRACT:
  1. Facts the user personally declared about themselves, their project, or their team
  2. Decisions the user made ("we're using X", "I prefer Y")
  3. Updates where the user explicitly overrides a prior statement
     ("actually, we switched to X", "no longer using Y")
  4. Concrete numbers, names, dates, or constraints the user provided

RULES — DO NOT EXTRACT:
  - Facts the assistant stated, suggested, or recommended
  - General knowledge about tools, frameworks, or technologies
    (e.g. "Redis is an in-memory store" — the LLM already knows this)
  - Opinions or assessments from the assistant ("that's a solid choice")
  - Hypothetical suggestions ("you could use X", "consider Y")
  - Restating or rephrasing what was already established
  - Team role assignments the assistant inferred (unless user confirmed)

STEP 1 — IDENTIFY THE SUBJECT
Use the most specific, resolvable name. Never a pronoun.
  1. Actual name if stated: "Raj", "FastAPI", "the auth module"
  2. Role/type if name unknown: "Database", "Backend Framework"
  3. First-person user claims → "USER"
  4. Pronouns (it, they, that) → resolve to most recent entity.
     If unresolvable → OMIT the belief entirely.

STEP 2 — NORMALISE THE VALUE
  - Numbers: digits only (5000 not "five thousand")
  - Currency: ISO code + amount (USD 5000 not "$5,000")
  - Dates: ISO 8601 (2024-03-15 not "March 15th")
  - Tech names: official capitalisation (PostgreSQL, FastAPI, TypeScript)
  - Port numbers: integer (8080 not "port 8080")
  - Status: snake_case (in_progress, not_started, done, blocked)

STEP 3 — CLASSIFY
  confidence: 0.95–1.0 direct statement, 0.75–0.90 clear implication,
              0.50–0.70 soft statement ("I think we will use...")

  belief_type:
    "assertion" — new fact stated for first time
    "update"    — explicitly replaces prior statement
                  triggers: "actually", "instead", "let's switch",
                  "changed", "no longer", "we decided on X instead"

  is_hypothetical: true if conditional or speculative
    triggers: "if", "might", "could", "as an option", "potentially",
              "we may consider", "in case", "if we face"
    IMPORTANT: Store hypotheticals — do NOT skip them.
    They are useful context. Flag them so they can be weighted lower.

  category: one of identity | technical | planning | constraint | state
    identity:   name, location, role, preference, biographical
    technical:  framework, database, language, tool, config, version
    planning:   task, assignment, deadline, dependency, milestone
    constraint: budget, limit, requirement, rule, must/cannot
    state:      current status, what is built, what was tried

  source_quote: verbatim excerpt from original text, MAX 100 chars.
    Trim to the key phrase. Never the full sentence.

OUTPUT FORMAT — return ONLY valid JSON array, no markdown, no explanation.
If no facts present, return [].
[
  {{
    "subject": "specific entity name — never a pronoun",
    "predicate": "the relation",
    "value": "normalised value",
    "confidence": 0.0,
    "belief_type": "assertion",
    "is_hypothetical": false,
    "category": "identity",
    "source": "user",
    "source_quote": "verbatim excerpt max 100 chars"
  }}
]

EXAMPLES:
Input: User: "I am Raj. Budget is $5k. Use FastAPI and PostgreSQL."
Output:
[
  {{"subject":"USER","predicate":"name is","value":"Raj","confidence":0.99,"belief_type":"assertion","is_hypothetical":false,"category":"identity","source":"user","source_quote":"I am Raj"}},
  {{"subject":"Project","predicate":"budget is","value":"USD 5000","confidence":0.97,"belief_type":"assertion","is_hypothetical":false,"category":"constraint","source":"user","source_quote":"Budget is $5k"}},
  {{"subject":"Backend Framework","predicate":"is","value":"FastAPI","confidence":0.97,"belief_type":"assertion","is_hypothetical":false,"category":"technical","source":"user","source_quote":"Use FastAPI"}},
  {{"subject":"Database","predicate":"is","value":"PostgreSQL","confidence":0.97,"belief_type":"assertion","is_hypothetical":false,"category":"technical","source":"user","source_quote":"and PostgreSQL"}}
]

Input: Assistant: "PostgreSQL is a popular database. You could also use Redis for caching."
Output: []
(Assistant suggestions — not user-declared facts)

Input: User: "Actually switch from PostgreSQL to SQLite."
Output:
[{{"subject":"Database","predicate":"is","value":"SQLite","confidence":0.97,"belief_type":"update","is_hypothetical":false,"category":"technical","source":"user","source_quote":"switch from PostgreSQL to SQLite"}}]

Input: User: "That sounds great!"
Output: []

Conversation to extract from:
{conversation}
"""

DEFAULT_EXTRACT_USER_PROMPT = DEFAULT_EXTRACT_PROMPT

DEFAULT_EXTRACT_ASSISTANT_PROMPT = DEFAULT_EXTRACT_PROMPT

DEFAULT_JUDGE_PROMPT = """Analyze the relationship between these two claims.

Premise: {premise}
Hypothesis: {hypothesis}

Determine if the hypothesis:
1. **contradicts** the premise (they cannot both be true)
2. **entails** the premise (if true, the premise is also true - they're semantically equivalent or the hypothesis is more specific)
3. is **neutral** (no clear relationship)

CRITICAL: Detect semantic equivalence carefully. For example:
- "USER likes Python" and "USER enjoys Python" = ENTAILMENT (same meaning)
- "USER lives in Paris" and "USER lives in France" = ENTAILMENT (specific location is Paris, which is in France)
- "USER has 5000 dollars" and "USER has USD 5000" = ENTAILMENT (same meaning, different phrasing)

Return ONLY a JSON object with keys 'relationship' ('contradiction', 'entailment', or 'neutral'), 'score' (float between 0.0 and 1.0 representing confidence), and 'reason' (string explanation).
Do not wrap in markdown.

Format:
{{"relationship": "entailment", "score": 0.95, "reason": "Both claims express the same preference for Python using different vocabulary"}}
"""


class TrackerConfig(BaseModel):
    """Configuration for the BeliefTracker."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Store settings
    store_type: str = Field(
        default="sqlite",
        description="Type of storage to use ('sqlite', 'redis', 'postgres').",
    )
    store_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the store."
    )

    @field_validator("store_type")
    @classmethod
    def validate_store_type(cls, v: str) -> str:
        valid = {"sqlite", "redis", "postgres"}
        if v.lower() not in valid:
            raise ValueError(f"store_type must be one of {valid}, got '{v}'")
        return v.lower()

    @field_validator("resolution_strategy")
    @classmethod
    def validate_resolution_strategy(cls, v: str) -> str:
        valid = {"overwrite", "keep_old", "raise"}
        if v not in valid:
            raise ValueError(f"resolution_strategy must be one of {valid}, got '{v}'")
        return v

    # Detection settings
    similarity_threshold: float = Field(
        default=0.82, description="Threshold for embedding similarity."
    )
    contradiction_threshold: float = Field(
        default=0.70, description="Threshold for finding contradictions."
    )
    entailment_threshold: float = Field(
        default=0.85,
        description="Threshold for detecting semantic entailment (belief duplication). If new belief is entailed by existing belief with score >= this threshold, skip the new belief.",
    )

    # Prompts
    extract_prompt_template: str = Field(
        default=DEFAULT_EXTRACT_PROMPT, description="Prompt used to extract beliefs."
    )
    extract_user_prompt_template: str = Field(
        default=DEFAULT_EXTRACT_USER_PROMPT,
        description="Prompt used to extract beliefs from user messages.",
    )
    extract_assistant_prompt_template: str = Field(
        default=DEFAULT_EXTRACT_ASSISTANT_PROMPT,
        description="Prompt used to extract beliefs from assistant messages.",
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
    internal_provider: Optional[Any] = Field(
        default=None, description="Explicit provider for tracker's internal LLM calls."
    )
    embed_provider: Optional[Any] = Field(
        default=None, description="Explicit provider for embedding generation."
    )
    embed_model: Optional[str] = Field(
        default=None, description="Model name to use for embeddings."
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

    # Contradiction resolution
    resolution_strategy: str = Field(
        default="overwrite",
        description="How to handle contradictions: 'overwrite' (replace old), 'keep_old' (ignore new), 'raise' (throw error).",
    )

    # Belief storage limits
    max_beliefs: int = Field(
        default=50,
        description="Maximum number of beliefs to store per session. New beliefs beyond this limit trigger eviction of lowest-confidence beliefs.",
    )
    belief_sort_strategy: str = Field(
        default="confidence_recency",
        description="Strategy for selecting top N beliefs: 'confidence_recency' (high confidence + recent turns) | 'recency' (most recent turns) | 'confidence' (highest confidence)",
    )

    # Belief TTL settings
    enable_belief_ttl: bool = Field(
        default=False,
        description="Enable automatic pruning of old beliefs based on age.",
    )
    belief_max_age_seconds: int = Field(
        default=86400,
        description="Maximum age in seconds for a belief before pruning (only if enable_belief_ttl=True).",
    )
    belief_ttl_check_interval: int = Field(
        default=3600,
        description="How often (in seconds) to check for expired beliefs in SQLite.",
    )

    # Staleness scoring for session resumption
    enable_staleness_scoring: bool = Field(
        default=True,
        description="Enable staleness scoring to deprioritize old beliefs during session resumption.",
    )
    staleness_threshold: float = Field(
        default=0.1,
        description="Minimum staleness score (confidence / days_since_referenced) to inject a belief. Beliefs below this threshold are excluded.",
    )

    # Token-aware belief injection
    enable_token_aware_injection: bool = Field(
        default=True,
        description="Enable token-aware belief injection for very long conversations.",
    )
    belief_budget_tokens: int = Field(
        default=300,
        description="Maximum tokens reserved for belief injection in prompts. If belief summary exceeds this, use relevance-based filtering.",
    )

    # Context injection filtering
    exclude_sources: list = Field(
        default_factory=lambda: ["assistant"],
        description="Belief sources to exclude from context injection (e.g. ['assistant'] to skip LLM-generated beliefs).",
    )
    min_injection_confidence: float = Field(
        default=0.80,
        description="Minimum confidence for a belief to be injected into context prompts.",
    )
    include_hypothetical_in_context: bool = Field(
        default=False,
        description="Whether to include hypothetical beliefs in context injection.",
    )

    # Judge timeout
    judge_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for LLM judge contradiction checks.",
    )

    # Confidence caps by source
    user_confidence_cap: float = Field(
        default=0.99,
        description="Maximum confidence for beliefs extracted from user messages.",
    )
    assistant_confidence_cap: float = Field(
        default=0.85,
        description="Maximum confidence for beliefs extracted from assistant responses.",
    )
