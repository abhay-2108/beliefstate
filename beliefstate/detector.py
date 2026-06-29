import logging
import math
import re
import unicodedata
from enum import Enum
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass
from beliefstate.config import TrackerConfig
from beliefstate.models import Belief
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.store.base import Store

logger = logging.getLogger(__name__)


# --- Outcome types ---


class Outcome(str, Enum):
    NEW = "new"
    DUPLICATE = "duplicate"
    CONTRADICTION = "contradiction"


@dataclass
class DetectionResult:
    belief: Belief
    outcome: Outcome
    matched_belief: Optional[Belief] = None
    score: float = 0.0
    reason: str = ""


# --- Negation detection ---

NEGATION_TOKENS = {
    "not",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "can't",
    "cannot",
    "shouldn't",
    "couldn't",
    "wouldn't",
    "never",
    "no",
    "none",
    "nobody",
    "nothing",
    "nowhere",
    "stopped",
    "quit",
    "unlike",
    "no longer",
    "not any",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
}


def has_negation(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    if "not " in text_lower or " not" in text_lower:
        return True
    if "n't" in text_lower:
        return True
    # Multi-word negation tokens: use substring match (no word boundaries)
    _MULTI_WORD_NEGATION = {"no longer", "not any"}
    for token in _MULTI_WORD_NEGATION:
        if token in text_lower:
            return True
    # Single-word negation tokens: use word-boundary match
    _SINGLE_WORD_NEGATION = {
        "not",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "can't",
        "cannot",
        "shouldn't",
        "couldn't",
        "wouldn't",
        "never",
        "no",
        "none",
        "nobody",
        "nothing",
        "nowhere",
        "stopped",
        "quit",
        "unlike",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
    }
    for token in _SINGLE_WORD_NEGATION:
        pattern = r"\b" + re.escape(token) + r"\b"
        if re.search(pattern, text_lower):
            return True
    return False


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    if len(v1) != len(v2):
        logger.warning(
            f"Embedding dimension mismatch: {len(v1)} vs {len(v2)}. "
            "Returning 0.0 similarity."
        )
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


def normalize_value(value: str) -> str:
    """Normalise value for exact-match dedup comparison (NFKC, lowercase, strip punctuation)."""
    v = unicodedata.normalize("NFKC", value.lower().strip())
    v = re.sub(r"[^\w\s]", "", v)
    v = re.sub(r"\s+", " ", v)
    return v


class ContradictionDetector:
    def __init__(
        self,
        adapter: ProviderAdapter,
        store: Store,
        config: TrackerConfig,
        judge: Optional[Any] = None,
    ):
        self.adapter = adapter
        self.store = store
        self.config = config
        if judge:
            self.judge = judge
        else:
            from beliefstate.judge import LLMJudge

            self.judge = LLMJudge(adapter, config)

    async def detect(
        self, session_id: str, new_beliefs: List[Belief]
    ) -> List[Tuple[Belief, Belief, float, str]]:
        """Detect contradictions between new beliefs and existing store."""
        contradictions = []

        for new_b in new_beliefs:
            if not new_b.embedding:
                continue

            matched_beliefs = await self.store.search_beliefs(
                session_id=session_id,
                embedding=new_b.embedding,
                threshold=self.config.similarity_threshold,
                limit=5,
            )

            for old_b in matched_beliefs:
                if (
                    old_b.embedding_model
                    and new_b.embedding_model
                    and old_b.embedding_model != new_b.embedding_model
                ):
                    logger.warning(
                        f"Embedding model mismatch: old='{old_b.embedding_model}' "
                        f"vs new='{new_b.embedding_model}'. "
                        f"Belief may be from a different embedding version. "
                        f"Skipping comparison."
                    )
                    continue

                is_contradiction, score, reason = await self.judge.check(old_b, new_b)
                if is_contradiction:
                    contradictions.append((old_b, new_b, score, reason))

        return contradictions

    async def detect_with_deduplication(
        self, session_id: str, new_beliefs: List[Belief]
    ) -> Tuple[List[Tuple[Belief, Belief, float, str]], List[Belief]]:
        """Detect contradictions AND deduplicate entailed beliefs.

        Step 0: Exact duplicate check (O(1), no LLM/embedding cost)
        Step 1: Embedding dimension version guard
        Step 2: Cosine similarity gate
        Step 3: NLI judgment on candidates
        """
        contradictions = []
        duplicates_to_skip = []

        for new_b in new_beliefs:
            if not new_b.embedding:
                continue

            # Step 0: Exact duplicate check — O(1), no LLM/embedding cost
            existing = await self.store.get_by_key(
                (new_b.subject or "").lower(),
                (new_b.predicate or "").lower(),
                session_id,
            )
            if existing and normalize_value(existing.value) == normalize_value(
                new_b.value
            ):
                if new_b not in duplicates_to_skip:
                    duplicates_to_skip.append(new_b)
                continue

            new_b_text = f"{new_b.predicate} {new_b.value}"
            has_new_negation = has_negation(new_b_text)

            matched_beliefs = await self.store.search_beliefs(
                session_id=session_id,
                embedding=new_b.embedding,
                threshold=self.config.similarity_threshold,
                limit=5,
            )

            for old_b in matched_beliefs:
                # Step 1: Embedding model version guard
                if (
                    old_b.embedding_model
                    and new_b.embedding_model
                    and old_b.embedding_model != new_b.embedding_model
                ):
                    logger.warning(
                        f"Embedding model mismatch: old='{old_b.embedding_model}' "
                        f"vs new='{new_b.embedding_model}'. "
                        f"Skipping vector comparison, using LLM judge instead."
                    )
                    is_contradiction, score, reason = await self.judge.check(
                        old_b, new_b
                    )
                    if is_contradiction:
                        contradictions.append((old_b, new_b, score, reason))
                    elif (
                        reason
                        and "entailment" in reason.lower()
                        and score >= self.config.entailment_threshold
                    ):
                        duplicates_to_skip.append(new_b)
                    continue

                # Guard against embedding dimension mismatch
                if (
                    old_b.embedding_dim
                    and new_b.embedding_dim
                    and old_b.embedding_dim != new_b.embedding_dim
                ):
                    logger.warning(
                        f"Embedding dimension mismatch for '{old_b.subject}': "
                        f"old={old_b.embedding_dim}D vs "
                        f"new={new_b.embedding_dim}D. "
                        f"Skipping vector comparison, using LLM judge instead."
                    )
                    is_contradiction, score, reason = await self.judge.check(
                        old_b, new_b
                    )
                    if is_contradiction:
                        contradictions.append((old_b, new_b, score, reason))
                    elif (
                        reason
                        and "entailment" in reason.lower()
                        and score >= self.config.entailment_threshold
                    ):
                        duplicates_to_skip.append(new_b)
                    continue

                # Check for negation — bypass cosine gate if found
                old_b_text = f"{old_b.predicate} {old_b.value}"
                has_old_negation = has_negation(old_b_text)

                if has_new_negation or has_old_negation:
                    logger.debug(
                        f"Negation detected in belief "
                        f"(new: {has_new_negation}, old: {has_old_negation}). "
                        f"Bypassing cosine similarity gate, using LLM judge."
                    )
                    is_contradiction, score, reason = await self.judge.check(
                        old_b, new_b
                    )
                    if is_contradiction:
                        contradictions.append((old_b, new_b, score, reason))
                    elif (
                        reason
                        and "entailment" in reason.lower()
                        and score >= self.config.entailment_threshold
                    ):
                        duplicates_to_skip.append(new_b)
                    continue

                # Normal path: cosine similarity check
                is_contradiction, score, reason = await self.judge.check(old_b, new_b)

                if is_contradiction:
                    contradictions.append((old_b, new_b, score, reason))

                elif score >= self.config.entailment_threshold:
                    if reason and "entailment" in reason.lower():
                        duplicates_to_skip.append(new_b)
                        break

        return contradictions, duplicates_to_skip
