import json
import logging
from typing import List
from beliefstate.config import TrackerConfig
from beliefstate.call import LLMCall
from beliefstate.models import Belief
from beliefstate.adapters.base import ProviderAdapter

logger = logging.getLogger(__name__)

class BeliefExtractor:
    def __init__(self, adapter: ProviderAdapter, config: TrackerConfig):
        self.adapter = adapter
        self.config = config
        
    async def extract(self, response_text: str, turn: int, source: str = "assistant") -> List[Belief]:
        """Extract factual claims from text using the LLM adapter."""
        prompt = self.config.extract_prompt_template.format(response=response_text)
        
        # We make an internal LLM call to extract beliefs
        call = LLMCall(messages=[{"role": "user", "content": prompt}])
        
        from pydantic import RootModel
        class BeliefListSchema(RootModel[List[Belief]]):
            pass
            
        try:
            llm_resp = await self.adapter.generate(call, response_format=BeliefListSchema)
            
            # Parse JSON robustly
            text = llm_resp.text.strip()
            data = None
            try:
                data = json.loads(text)
            except Exception:
                import re
                match_arr = re.search(r'\[.*\]', text, re.DOTALL)
                match_obj = re.search(r'\{.*\}', text, re.DOTALL)
                if match_obj:
                    try:
                        data = json.loads(match_obj.group(0))
                    except Exception:
                        pass
                if data is None and match_arr:
                    try:
                        data = json.loads(match_arr.group(0))
                    except Exception:
                        pass
                        
            if data is None:
                return []
                
            if isinstance(data, dict):
                data = data.get("beliefs", data.get("root", []))
                if not isinstance(data, list):
                    data = []
            
            beliefs = []
            temp_beliefs = []
            from pydantic import ValidationError
            for item in data:
                try:
                    subj = item.get("subject", "").strip()
                    subj_upper = subj.upper()
                    
                    # Deterministic Post-Processing Fallback
                    if source == "user":
                        if subj_upper in ["I", "ME", "MY", "MINE", "MYSELF"]:
                            subj = "USER"
                        elif subj_upper in ["YOU", "YOUR", "YOURS", "YOURSELF"]:
                            subj = "ASSISTANT"
                    elif source == "assistant":
                        if subj_upper in ["I", "ME", "MY", "MINE", "MYSELF"]:
                            subj = "ASSISTANT"
                        elif subj_upper in ["YOU", "YOUR", "YOURS", "YOURSELF"]:
                            subj = "USER"

                    b = Belief(
                        subject=subj,
                        predicate=item.get("predicate", ""),
                        value=item.get("value", ""),
                        confidence=float(item.get("confidence", 1.0)),
                        turn=turn,
                        source=source
                    )
                    temp_beliefs.append(b)
                except ValidationError as ve:
                    logger.warning(f"Skipping malformed belief: {ve}")
                    
            if temp_beliefs:
                try:
                    # Request batch embeddings for all valid beliefs in a single API call
                    texts_to_embed = [f"{b.subject} {b.predicate} {b.value}" for b in temp_beliefs]
                    embeddings = await self.adapter.get_embeddings(texts_to_embed)
                    for b, emb in zip(temp_beliefs, embeddings):
                        b.embedding = emb
                        beliefs.append(b)
                except Exception as e:
                    logger.warning(f"Error enqueuing batch embeddings: {e}. Falling back to individual embedding generation.")
                    # Fallback to individual embedding requests to preserve the extracted facts
                    for b in temp_beliefs:
                        try:
                            text_to_embed = f"{b.subject} {b.predicate} {b.value}"
                            b.embedding = await self.adapter.get_embedding(text_to_embed)
                            beliefs.append(b)
                        except Exception as ie:
                            logger.error(f"Error embedding individual belief fallback: {ie}", exc_info=True)
                            
            return beliefs
        except Exception as e:
            # Silently fail or log for extraction errors so we don't crash the main app
            logger.error(f"Belief extraction error: {e}", exc_info=True)
            return []
