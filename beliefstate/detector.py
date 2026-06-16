import json
import logging
import math
from typing import List, Tuple
from beliefstate.config import TrackerConfig
from beliefstate.models import Belief
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.store.base import Store
from beliefstate.call import LLMCall

logger = logging.getLogger(__name__)

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not v1 or not v2: 
        return 0.0
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(b*b for b in v2))
    if mag1 == 0.0 or mag2 == 0.0: 
        return 0.0
    return dot / (mag1 * mag2)

class ContradictionDetector:
    def __init__(self, adapter: ProviderAdapter, store: Store, config: TrackerConfig):
        self.adapter = adapter
        self.store = store
        self.config = config
        
    async def detect(self, session_id: str, new_beliefs: List[Belief]) -> List[Tuple[Belief, Belief, float, str]]:
        """Detect contradictions between new beliefs and existing store."""
        contradictions = []
        
        for new_b in new_beliefs:
            if not new_b.embedding:
                continue
                
            # Database-side vector search for top candidate beliefs
            matched_beliefs = await self.store.search_beliefs(
                session_id=session_id,
                embedding=new_b.embedding,
                threshold=self.config.similarity_threshold,
                limit=5
            )
            
            for old_b in matched_beliefs:
                sim = cosine_similarity(new_b.embedding, old_b.embedding)
                
                # If semantically related, do a deep check with LLM NLI
                premise = f"{old_b.subject} {old_b.predicate} {old_b.value}"
                hypothesis = f"{new_b.subject} {new_b.predicate} {new_b.value}"
                
                prompt = self.config.judge_prompt_template.format(
                    premise=premise, hypothesis=hypothesis
                )
                
                call = LLMCall(messages=[{"role": "user", "content": prompt}])
                from pydantic import BaseModel, Field
                class ContradictionResolution(BaseModel):
                    relationship: str = Field(description="The relationship between premise and hypothesis: 'contradiction', 'entailment', or 'neutral'")
                    score: float = Field(description="Confidence score between 0.0 and 1.0")
                    reason: str = Field(description="Explanation of the relationship decision")
                    
                try:
                    llm_resp = await self.adapter.generate(call, response_format=ContradictionResolution)
                    raw_text = llm_resp.text.strip()
                    data = None
                    try:
                        data = json.loads(raw_text)
                    except Exception:
                        import re
                        match_obj = re.search(r'\{.*\}', raw_text, re.DOTALL)
                        if match_obj:
                            try:
                                data = json.loads(match_obj.group(0))
                            except Exception:
                                pass
                                
                    if data is None:
                        continue
                        
                    if data.get("relationship") == "contradiction":
                        score = float(data.get("score", 0.0))
                        if score >= self.config.contradiction_threshold:
                            contradictions.append((old_b, new_b, score, data.get("reason", "")))
                            
                except Exception as e:
                    logger.error(f"Contradiction detection error: {e}", exc_info=True)
                        
        return contradictions
