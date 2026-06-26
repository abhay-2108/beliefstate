import json
import asyncio
import logging
from typing import Tuple, Protocol, runtime_checkable, Optional, Any
from beliefstate.models import Belief
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.config import TrackerConfig
from beliefstate.call import LLMCall

logger = logging.getLogger("beliefstate.judge")


@runtime_checkable
class ContradictionJudge(Protocol):
    """Protocol for checking semantic contradictions between beliefs."""

    async def check(self, old: Belief, new: Belief) -> Tuple[bool, float, str]:
        """
        Evaluate if a new belief contradicts an old belief.

        Returns:
            Tuple[is_contradiction (bool), confidence (float), reason (str)]
        """
        ...


class LLMJudge(ContradictionJudge):
    """Contradiction judge using an LLM provider adapter."""

    def __init__(self, adapter: ProviderAdapter, config: TrackerConfig):
        self.adapter = adapter
        self.config = config

    async def check(self, old: Belief, new: Belief) -> Tuple[bool, float, str]:
        premise = f"{old.subject} {old.predicate} {old.value}"
        hypothesis = f"{new.subject} {new.predicate} {new.value}"

        prompt = self.config.judge_prompt_template.format(
            premise=premise, hypothesis=hypothesis
        )

        call = LLMCall(messages=[{"role": "user", "content": prompt}])
        from pydantic import BaseModel, Field

        class ContradictionResolution(BaseModel):
            relationship: str = Field(
                description="The relationship between premise and hypothesis: 'contradiction', 'entailment', or 'neutral'"
            )
            score: float = Field(description="Confidence score between 0.0 and 1.0")
            reason: str = Field(description="Explanation of the relationship decision")

        try:
            llm_resp = await asyncio.wait_for(
                self.adapter.generate(call, response_format=ContradictionResolution),
                timeout=self.config.judge_timeout,
            )
            raw_text = llm_resp.text.strip()
            data = None
            try:
                data = json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                import re

                match_obj = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if match_obj:
                    try:
                        data = json.loads(match_obj.group(0))
                    except json.JSONDecodeError as e:
                        logger.debug(f"Judge JSON parse failed: {e}")

            if data is None:
                return False, 0.0, "Failed to parse judge response"

            relationship = data.get("relationship", "neutral")
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "")

            if (
                relationship == "contradiction"
                and score >= self.config.contradiction_threshold
            ):
                return True, score, reason
            return False, score, reason
        except Exception as e:
            logger.error(f"LLM judge error checking contradiction: {e}", exc_info=True)
            return False, 0.0, str(e)


class LocalNLIJudge(ContradictionJudge):
    """Contradiction judge using a local HuggingFace NLI Cross-Encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        threshold: float = 0.7,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self._pipeline: Optional[Any] = None

    def _init_pipeline(self) -> None:
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError(
                    "transformers and torch are required to run LocalNLIJudge. "
                    'Please install them using `pip install "beliefstate[local]"` '
                    "or `pip install transformers torch`."
                )
            self._pipeline = pipeline("text-classification", model=self.model_name)

    async def check(self, old: Belief, new: Belief) -> Tuple[bool, float, str]:
        self._init_pipeline()

        premise = f"{old.subject} {old.predicate} {old.value}"
        hypothesis = f"{new.subject} {new.predicate} {new.value}"

        import asyncio

        loop = asyncio.get_running_loop()

        pipeline_fn = self._pipeline
        if pipeline_fn is None:
            return False, 0.0, "Pipeline not initialized"

        try:
            # Run inference in a threadpool executor to avoid blocking the event loop
            res = await loop.run_in_executor(
                None, lambda: pipeline_fn({"text": premise, "text_pair": hypothesis})
            )

            if not res or not isinstance(res, list) or len(res) == 0:
                return False, 0.0, "Empty pipeline response"

            label = str(res[0].get("label", "")).lower()
            score = float(res[0].get("score", 0.0))

            # Map label names
            is_contra = "contradiction" in label or label == "label_0" or label == "0"

            if is_contra and score >= self.threshold:
                return (
                    True,
                    score,
                    f"Local NLI classification: {label} (score: {score:.2f})",
                )
            return (
                False,
                score,
                f"Local NLI classification: {label} (score: {score:.2f})",
            )
        except Exception as e:
            logger.error(
                f"Local NLI judge error checking contradiction: {e}", exc_info=True
            )
            return False, 0.0, str(e)
