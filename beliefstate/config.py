from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict

DEFAULT_EXTRACT_PROMPT = """You are an expert belief state extractor. From this text, extract all factual claims being asserted.

CRITICAL RULES FOR SUBJECT NORMALIZATION:
1. First-person pronouns (I, me, my, mine, we, us) MUST be mapped to "USER".
2. Second-person pronouns (you, your, yours) referring to the AI MUST be mapped to "ASSISTANT".
3. Never use the user's name as the subject for their own traits; always map it to "USER".
4. Resolve ambiguous pronouns to their actual entities when possible:
   - "It" -> refer back to the most recent noun mentioned (the thing being discussed)
   - "They" -> refer back to the most recent plural noun
   - Example: "I like Python. It's great." -> Extract as "USER likes Python" and "Python is great" (NOT "USER likes it")
   - Example: "Sarah and John are friends. They work together." -> Extract as "Sarah and John are friends" and "Sarah and John work together" (resolve "they" to the actual people/entity)

CRITICAL RULES FOR VALUE NORMALIZATION:
Apply these normalization rules to all extracted values to ensure consistency:

NUMBERS:
- Always use digits (not words). Example: "5000" not "five thousand"
- Remove commas/spaces from large numbers. Example: "5000" not "5,000"
- For decimals, use standard notation. Example: "3.14" not "3.14159265"

CURRENCY:
- Always use ISO 4217 currency codes. Example: "USD 5000" not "$5,000" or "five thousand dollars"
- Format: "[CURRENCY_CODE] [amount]" e.g., "USD 5000", "EUR 100", "GBP 50"

DATES:
- Always use ISO 8601 format (YYYY-MM-DD). Example: "2024-03-15" not "March 15" or "15/03/2024"
- If only year and month known: "2024-03"
- If only year known: "2024"

PERCENTAGES:
- Use decimal notation (0.0 to 1.0). Example: "0.15" for 15%, not "15%" or "0.15%"

CRITICAL RULES FOR BELIEF_TYPE:
- Set belief_type="assertion" for normal factual claims (default).
- Set belief_type="update" for explicit temporal changes. Examples:
  * "I changed my mind about Python" (previous: "likes Python", new: "doesn't like Python")
  * "I no longer work at Google" (previous: "works at Google", new: different employer)
  * "Actually, I meant to say..." or "I misspoke earlier about..."
  * "My previous answer was wrong, the correct..."
  * Any statement containing: "used to", "previously", "at first I thought", "I was wrong about", "changing my stance on"
- Temporal markers that trigger belief_type="update":
  * Correction words: wrong, incorrect, mistaken, misspoke, misstated, correction
  * Change words: changed, switched, shift, pivot, reconsider, rethink, revise
  * Time words: used to, previously, at first, initially, before, originally, now I realize
  * Negation of previous: "I don't" (after "I do"), "I hate" (after "I love")

CRITICAL RULES FOR IS_HYPOTHETICAL:
- Set is_hypothetical=true for beliefs that are conditional, speculative, or example-based.
- These beliefs should NOT be injected into system prompts (they're exploratory, not factual).
- Markers for hypothetical beliefs:
  * Conditional: "If I were...", "If I had...", "Suppose I...", "Let's say I..."
  * Speculative: "I might...", "I could...", "I'd probably...", "I guess I might..."
  * Example/illustration: "For example, if...", "Say I...", "Imagine I...", "What if I..."
  * Roleplaying: "In this scenario...", "Playing the role of...", "As if I were..."
  * Hypothetical comparison: "If I were like...", "Compared to if I had..."

Examples for is_hypothetical:
Input: "If I had a million dollars, I'd buy a house in Paris"
Output: is_hypothetical=true (conditional speculation)

Input: "Imagine I worked at Google - I'd probably..."
Output: is_hypothetical=true (hypothetical scenario)

Input: "I actually work at Google right now"
Output: is_hypothetical=false (factual assertion)

EXAMPLES:
Input: "I have $5,000 and was born on March 15th, 1990"
Output: 
[
  {{"subject": "USER", "predicate": "has", "value": "USD 5000", "confidence": 1.0, "belief_type": "assertion", "source": "user"}},
  {{"subject": "USER", "predicate": "born on", "value": "1990-03-15", "confidence": 1.0, "belief_type": "assertion", "source": "user"}}
]

Input: "I used to work at Microsoft but now I work at Google"
Output: 
[
  {{"subject": "USER", "predicate": "works at", "value": "Google", "confidence": 1.0, "belief_type": "update", "source": "user"}}
]

Input: "I like Python. It's a powerful language."
Output: 
[
  {{"subject": "USER", "predicate": "likes", "value": "Python", "confidence": 1.0, "belief_type": "assertion", "source": "user"}},
  {{"subject": "Python", "predicate": "is", "value": "powerful", "confidence": 0.95, "belief_type": "assertion", "source": "user"}}
]

Input: "Actually, I was wrong about Python. I don't like it."
Output: 
[
  {{"subject": "USER", "predicate": "likes", "value": "Python", "confidence": 0.0, "belief_type": "update", "source": "user"}}
]

Return ONLY a JSON array of objects with this format:
[
  {{"subject": "...", "predicate": "...", "value": "...", "confidence": 0.0-1.0, "belief_type": "assertion"|"update", "source": "..."}}
]

If no claims, return [].

Text:
{response}
"""

DEFAULT_EXTRACT_USER_PROMPT = DEFAULT_EXTRACT_PROMPT

DEFAULT_EXTRACT_ASSISTANT_PROMPT = """You are an expert belief state extractor. From this text, extract all factual claims being asserted.

CRITICAL RULES FOR SUBJECT NORMALIZATION:
1. First-person pronouns (I, me, my, mine, we, us) referring to the AI MUST be mapped to "ASSISTANT".
2. Second-person pronouns (you, your, yours) referring to the user MUST be mapped to "USER".
3. Never use the user's name as the subject for their own traits; always map it to "USER".
4. Resolve ambiguous pronouns to their actual entities when possible:
   - "It" -> refer back to the most recent noun mentioned (the thing being discussed)
   - "They" -> refer back to the most recent plural noun
   - Example: "I run on servers in Paris. It's a cluster." -> Extract as "ASSISTANT runs on servers in Paris" and "servers in Paris is a cluster" (resolve "it" and "I")

CRITICAL RULES FOR VALUE NORMALIZATION:
Apply these normalization rules to all extracted values to ensure consistency:

NUMBERS:
- Always use digits (not words). Example: "5000" not "five thousand"
- Remove commas/spaces from large numbers. Example: "5000" not "5,000"
- For decimals, use standard notation. Example: "3.14" not "3.14159265"

CURRENCY:
- Always use ISO 4217 currency codes. Example: "USD 5000" not "$5,000" or "five thousand dollars"
- Format: "[CURRENCY_CODE] [amount]" e.g., "USD 5000", "EUR 100", "GBP 50"

DATES:
- Always use ISO 8601 format (YYYY-MM-DD). Example: "2024-03-15" not "March 15" or "15/03/2024"
- If only year and month known: "2024-03"
- If only year known: "2024"

PERCENTAGES:
- Use decimal notation (0.0 to 1.0). Example: "0.15" for 15%, not "15%" or "0.15%"

CRITICAL RULES FOR BELIEF_TYPE:
- Set belief_type="assertion" for normal factual claims (default).
- Set belief_type="update" for explicit temporal changes. Examples:
  * "I changed my mind about supporting v2" (previous: "supports v2", new: "does not support v2")
  * "I no longer run on AWS" (previous: "runs on AWS", new: different cloud)
- Temporal markers that trigger belief_type="update":
  * Correction words: wrong, incorrect, mistaken, misspoke, misstated, correction
  * Change words: changed, switched, shift, pivot, reconsider, rethink, revise
  * Time words: used to, previously, at first, initially, before, originally, now I realize

CRITICAL RULES FOR IS_HYPOTHETICAL:
- Set is_hypothetical=true for beliefs that are conditional, speculative, or example-based.
- These beliefs should NOT be injected into system prompts (they're exploratory, not factual).
- Markers for hypothetical beliefs:
  * Conditional: "If I were...", "If I had...", "Suppose I...", "Let's say I..."
  * Speculative: "I might...", "I could...", "I'd probably...", "I guess I might..."
  * Example/illustration: "For example, if...", "Say I...", "Imagine I...", "What if I..."
  * Roleplaying: "In this scenario...", "Playing the role of...", "As if I were..."

EXAMPLES:
Input: "I run on servers maintained by Alibaba Cloud"
Output: 
[
  {{"subject": "ASSISTANT", "predicate": "runs on", "value": "servers maintained by Alibaba Cloud", "confidence": 1.0, "belief_type": "assertion", "source": "assistant"}}
]

Input: "I used to support v2 but now I only support v3"
Output: 
[
  {{"subject": "ASSISTANT", "predicate": "supports", "value": "v3", "confidence": 1.0, "belief_type": "update", "source": "assistant"}}
]

Input: "I can help you with Python. It's a powerful language."
Output: 
[
  {{"subject": "ASSISTANT", "predicate": "can help", "value": "USER with Python", "confidence": 1.0, "belief_type": "assertion", "source": "assistant"}},
  {{"subject": "Python", "predicate": "is", "value": "powerful", "confidence": 0.95, "belief_type": "assertion", "source": "assistant"}}
]

Input: "Actually, I was wrong about that feature. I don't support it."
Output: 
[
  {{"subject": "ASSISTANT", "predicate": "supports", "value": "that feature", "confidence": 0.0, "belief_type": "update", "source": "assistant"}}
]

Return ONLY a JSON array of objects with this format:
[
  {{"subject": "...", "predicate": "...", "value": "...", "confidence": 0.0-1.0, "belief_type": "assertion"|"update", "source": "..."}}
]

If no claims, return [].

Text:
{response}
"""

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

    # Belief storage limits
    max_beliefs: int = Field(
        default=50,
        description="Maximum number of beliefs to inject into prompts (prevents context window overflow).",
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
        default=86400,  # 24 hours
        description="Maximum age in seconds for a belief before pruning (only if enable_belief_ttl=True).",
    )
    belief_ttl_check_interval: int = Field(
        default=3600,  # 1 hour
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
        default=500,
        description="Maximum tokens reserved for belief injection in prompts. If belief summary exceeds this, use relevance-based filtering.",
    )
