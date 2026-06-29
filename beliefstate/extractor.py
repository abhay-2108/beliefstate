import json
import logging
import re
from typing import Any, List, Optional
from beliefstate.config import TrackerConfig, DEFAULT_EXTRACT_PROMPT
from beliefstate.call import LLMCall
from beliefstate.models import Belief
from beliefstate.adapters.base import ProviderAdapter

logger = logging.getLogger(__name__)


def normalize_numbers(text: str) -> str:
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    return text


def normalize_currency(text: str) -> str:
    patterns = [
        (r"\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", r"USD \1"),
        (r"€\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", r"EUR \1"),
        (r"£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", r"GBP \1"),
        (r"¥\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", r"JPY \1"),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"(USD|EUR|GBP|JPY)\s+(\d+),(\d+)", r"\1 \2\3", text)
    return text


def normalize_dates(text: str) -> str:
    months = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }

    def replace_date_long(match: re.Match[str]) -> str:
        month_name = match.group(1).lower()
        day = re.sub(r"(st|nd|rd|th)", "", match.group(2)).zfill(2)
        year = match.group(3)
        month = months.get(month_name, "01")
        return f"{year}-{month}-{day}"

    text = re.sub(
        r"(" + "|".join(months.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})",
        replace_date_long,
        text,
        flags=re.IGNORECASE,
    )

    def replace_date_dmy(match: re.Match[str]) -> str:
        day = re.sub(r"(st|nd|rd|th)", "", match.group(1)).zfill(2)
        month_name = match.group(2).lower()
        year = match.group(3)
        month = months.get(month_name, "01")
        return f"{year}-{month}-{day}"

    text = re.sub(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(months.keys()) + r")\s+(\d{4})",
        replace_date_dmy,
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(\d{1,2})/(\d{1,2})/(\d{4})",
        lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}",
        text,
    )
    return text


def normalize_percentages(text: str) -> str:
    def replace_percent(match: re.Match[str]) -> str:
        value = float(match.group(1)) / 100
        return f"{value:.2f}".rstrip("0").rstrip(".")

    # Only match percentages followed by whitespace or end of string
    # to avoid false positives like version numbers (e.g., "v1.50%")
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%(?=\s|$)", replace_percent, text)
    return text


def normalize_value_format(value: str) -> str:
    if not value:
        return value
    value = normalize_numbers(value)
    value = normalize_currency(value)
    value = normalize_dates(value)
    value = normalize_percentages(value)
    return value


def classify_response_type(text: str) -> str:
    if not text:
        return "conversational"
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return "json"
    if text.startswith("{") and text.endswith("}"):
        try:
            json.loads(text)
            return "json"
        except json.JSONDecodeError:
            pass
    sql_keywords = [
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "ALTER",
        "DROP",
        "FROM",
        "WHERE",
        "JOIN",
        "GROUP BY",
        "ORDER BY",
    ]
    text_upper = text.split("\n")[0].upper()
    if any(keyword in text_upper for keyword in sql_keywords):
        return "sql"
    code_pattern = r"```[\w]*\n"
    if re.search(code_pattern, text):
        code_blocks = re.findall(r"```[\w]*\n.*?\n```", text, re.DOTALL)
        total_code_chars = sum(len(block) for block in code_blocks)
        if total_code_chars / len(text) > 0.5:
            return "code"
        else:
            return "markdown_heavy"
    if text.count("`") > 4:
        return "markdown_heavy"
    return "conversational"


def _is_trivial_response(text: str) -> bool:
    """Check if an assistant response is trivial and should be skipped for extraction."""
    if not text or not text.strip():
        return True
    text = text.strip()
    if len(text) < 20:
        trivial_patterns = [
            r"^(sure|ok|okay|got it|understood|great|see|thanks|thank you|yes|no|right|gotcha|will do|absolutely|definitely|of course|no problem|you're welcome|i see|noted|alright|fine|perfect|sounds good|will do|i'll do that|let me know|i understand|got it)\s*[!.]*$",
        ]
        text_lower = text.lower()
        for pattern in trivial_patterns:
            if re.match(pattern, text_lower):
                return True
    code_chars = set("{}()=;:|#|\\><")
    if text:
        code_count = sum(1 for c in text if c in code_chars)
        if code_count / len(text) > 0.6:
            return True
    if text.startswith("{") or text.startswith("["):
        return True
    return False


def chunk_response_by_paragraphs(text: str, max_chunk_length: int = 2000) -> List[str]:
    if not text or len(text) <= max_chunk_length:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    if not paragraphs:
        return [text]
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_length = len(para) + 2
        if current_chunk and current_length + para_length > max_chunk_length:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    if not chunks:
        return [text]
    logger.debug(f"Chunked response into {len(chunks)} paragraphs for extraction")
    return chunks


def recover_json_from_response(text: str) -> Optional[List[Any]]:
    if not text or not isinstance(text, str):
        return None
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if "beliefs" in data:
                b = data["beliefs"]
                return b if isinstance(b, list) else None
            if "root" in data:
                r = data["root"]
                return r if isinstance(r, list) else None
        return None
    except json.JSONDecodeError:
        pass

    text_clean = re.sub(r"```(?:json)?\s*\n?", "", text)
    text_clean = re.sub(r"```\s*\n?", "", text_clean)
    text_clean = text_clean.strip()

    try:
        data = json.loads(text_clean)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and ("beliefs" in data or "root" in data):
            result = data.get("beliefs") or data.get("root")
            return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        pass

    start_idx = text_clean.find("[")
    if start_idx == -1:
        return None
    bracket_count = 0
    end_idx = -1
    for i in range(start_idx, len(text_clean)):
        if text_clean[i] == "[":
            bracket_count += 1
        elif text_clean[i] == "]":
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i
                break
    if end_idx == -1:
        end_idx = text_clean.rfind("]")
    if end_idx == -1:
        return None
    json_substr = text_clean[start_idx : end_idx + 1]

    try:
        data = json.loads(json_substr)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    json_substr = json_substr.replace("\u201c", '"').replace("\u201d", '"')
    json_substr = json_substr.replace("'", '"')

    try:
        data = json.loads(json_substr)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    if not json_substr.rstrip().endswith("]"):
        json_substr_recover = json_substr.rstrip()
        if json_substr_recover.endswith(","):
            json_substr_recover = json_substr_recover[:-1]
        json_substr_recover += "]"
        try:
            data = json.loads(json_substr_recover)
            if isinstance(data, list):
                logger.debug("Recovered truncated JSON by adding closing bracket")
                return data
        except json.JSONDecodeError:
            pass

    if json_substr.count("{") > json_substr.count("}"):
        missing_braces = json_substr.count("{") - json_substr.count("}")
        json_substr_recover = json_substr.rstrip()
        if json_substr_recover.endswith(","):
            json_substr_recover = json_substr_recover[:-1]
        json_substr_recover += "}" * missing_braces + "]"
        try:
            data = json.loads(json_substr_recover)
            if isinstance(data, list):
                logger.debug("Recovered truncated JSON by adding missing braces")
                return data
        except json.JSONDecodeError:
            pass

    logger.warning(
        f"Failed to recover JSON from response. Last attempt: {json_substr[:100]}..."
    )
    return None


# --- Confidence calibration ---

HEDGING_PATTERNS = {
    r"might|may|could|perhaps|possibly": 0.60,
    r"think|believe|probably|likely": 0.70,
    r"want to|planning to|considering": 0.75,
    r"not sure|unsure|maybe": 0.50,
}


def calibrate_confidence(belief: Belief) -> Belief:
    """Post-extraction calibration: lower confidence and set is_hypothetical when
    the source_quote contains hedging patterns.

    Applies a floor of 0.10 to prevent confidence from reaching zero."""
    quote = getattr(belief, "source_quote", "").lower()
    if not quote:
        return belief
    for pattern, ceiling in HEDGING_PATTERNS.items():
        if re.search(pattern, quote):
            belief.confidence = max(0.10, min(belief.confidence, ceiling))
            if ceiling <= 0.60:
                belief.is_hypothetical = True
            break
    return belief


# --- Pydantic schema for structured output ---


class ExtractedBeliefSchema:
    """Lazy schema builder for extraction."""

    @staticmethod
    def build() -> type:
        from pydantic import BaseModel, Field, RootModel

        class ExtractedBelief(BaseModel):
            subject: str = Field(description="The entity or concept being described")
            predicate: str = Field(description="The relationship or action")
            value: str = Field(description="The target or property value")
            confidence: float = Field(default=1.0, ge=0.0, le=1.0)
            belief_type: str = Field(
                default="assertion", description="'assertion' or 'update'"
            )
            is_hypothetical: bool = Field(default=False)
            category: str = Field(
                default="", description="identity|technical|planning|constraint|state"
            )
            source: str = Field(default="user", description="user or assistant")
            source_quote: str = Field(
                default="", description="verbatim excerpt max 100 chars"
            )

        class BeliefListSchema(RootModel[List[ExtractedBelief]]):
            pass

        return BeliefListSchema


class BeliefExtractor:
    def __init__(self, adapter: ProviderAdapter, config: TrackerConfig):
        self.adapter = adapter
        self.config = config
        provider = config.embed_provider
        self.embedding_adapter: ProviderAdapter = (
            provider if provider is not None else adapter
        )
        self.embedding_model: str = (
            getattr(config, "embed_model", None)
            or getattr(self.embedding_adapter, "embed_model", "")
            or ""
        )
        self.embedding_dim = self._get_embedding_dim()

    def _get_embedding_dim(self) -> int:
        model_name = (self.embedding_model or "").lower()
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "nomic-embed-text": 768,
            "all-minilm-l6-v2": 384,
            "sentence-transformers/all-minilm-l6-v2": 384,
            "all-mpnet-base-v2": 768,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }
        for key, dim in dim_map.items():
            if key in model_name:
                return dim
        if self.embedding_adapter is not None and hasattr(
            self.embedding_adapter, "embedding_dim"
        ):
            return int(getattr(self.embedding_adapter, "embedding_dim"))
        return 0

    def _is_trivial(self, text: str) -> bool:
        """Check if assistant response is trivial (for pre-filter)."""
        return _is_trivial_response(text)

    async def process_turn(
        self,
        user_message: str,
        assistant_response: str,
        session_id: str,
        turn: int,
    ) -> List[Belief]:
        """Extract beliefs from BOTH user message and assistant response.

        Pre-filters trivial assistant responses but still extracts from user message.
        Applies confidence ceiling by source after extraction.
        """
        is_trivial = self._is_trivial(assistant_response)

        if is_trivial:
            combined_text = user_message
        else:
            combined_text = f"User: {user_message}\nAssistant: {assistant_response}"

        if not combined_text or not combined_text.strip():
            return []

        raw_beliefs = await self._call_extraction_llm(combined_text, turn)

        for b in raw_beliefs:
            if b.source == "assistant":
                b.confidence = min(b.confidence, self.config.assistant_confidence_cap)
            else:
                b.confidence = min(b.confidence, self.config.user_confidence_cap)
            b = calibrate_confidence(b)
            if not b.session_id:
                b.session_id = session_id

        return raw_beliefs

    async def _call_extraction_llm(self, text: str, turn: int) -> List[Belief]:
        """Call the LLM to extract beliefs from text."""
        prompt_template = self.config.extract_prompt_template
        prompt = prompt_template.format(conversation=text)

        call = LLMCall(messages=[{"role": "user", "content": prompt}])

        BeliefListSchema = ExtractedBeliefSchema.build()

        try:
            llm_resp = await self.adapter.generate(
                call, response_format=BeliefListSchema
            )
            raw_text = llm_resp.text.strip()
            data = recover_json_from_response(raw_text)

            if data is None:
                return []

            if isinstance(data, dict):
                data = data.get("beliefs", data.get("root", []))
                if not isinstance(data, list):
                    data = []

            beliefs = []
            temp_beliefs: list[Belief] = []

            for item in data:
                try:
                    subj = item.get("subject", "").strip()
                    subj_upper = subj.upper()
                    source = item.get("source", "user")

                    if subj_upper in ["I", "ME", "MY", "MINE", "MYSELF"]:
                        subj = "USER" if source == "user" else "ASSISTANT"
                    elif subj_upper in ["YOU", "YOUR", "YOURS", "YOURSELF"]:
                        subj = "ASSISTANT" if source == "user" else "USER"

                    source_quote = item.get("source_quote", "")
                    if source_quote and len(source_quote) > 100:
                        source_quote = source_quote[:100]

                    b = Belief(
                        subject=subj,
                        predicate=item.get("predicate", ""),
                        value=normalize_value_format(item.get("value", "")),
                        confidence=float(item.get("confidence", 1.0)),
                        turn=turn,
                        source=source,
                        source_quote=source_quote,
                        category=item.get("category", ""),
                        belief_type=item.get("belief_type", "assertion"),
                        is_hypothetical=item.get("is_hypothetical", False),
                    )
                    temp_beliefs.append(b)
                except Exception as ve:
                    logger.warning(f"Skipping malformed belief: {ve}")

            if temp_beliefs:
                try:
                    texts_to_embed = [
                        f"{b.subject} {b.predicate} {b.value}" for b in temp_beliefs
                    ]
                    embeddings = await self.embedding_adapter.get_embeddings(
                        texts_to_embed
                    )
                    for b, emb in zip(temp_beliefs, embeddings):
                        b.embedding = emb
                        b.embedding_model = self.embedding_model
                        b.embedding_dim = self.embedding_dim
                        beliefs.append(b)
                except Exception as e:
                    logger.warning(
                        f"Error enqueuing batch embeddings: {e}. "
                        f"Falling back to individual embedding generation."
                    )
                    for b in temp_beliefs:
                        try:
                            text_to_embed = f"{b.subject} {b.predicate} {b.value}"
                            b.embedding = await self.embedding_adapter.get_embedding(
                                text_to_embed
                            )
                            b.embedding_model = self.embedding_model
                            b.embedding_dim = self.embedding_dim
                            beliefs.append(b)
                        except Exception as ie:
                            logger.error(
                                f"Error embedding individual belief fallback: {ie}",
                                exc_info=True,
                            )

            return beliefs
        except Exception as e:
            logger.error(f"Belief extraction error: {e}", exc_info=True)
            return []

    async def extract(
        self, response_text: str, turn: int, source: str = "assistant"
    ) -> List[Belief]:
        """Legacy extract method — processes a single text block.

        For new code, prefer process_turn() which handles both user and assistant.
        """
        resp_type = classify_response_type(response_text)

        if resp_type in ["code", "json", "sql"]:
            logger.debug(f"Skipping extraction for {resp_type} response type")
            return []

        extraction_text = response_text
        if resp_type == "markdown_heavy":
            extraction_text = re.sub(
                r"```[\w]*\n.*?\n```", "", response_text, flags=re.DOTALL
            )
            extraction_text = extraction_text.strip()
            if not extraction_text:
                return []

        chunks = chunk_response_by_paragraphs(extraction_text, max_chunk_length=2000)

        all_beliefs = []
        for chunk_idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            if len(chunks) > 1:
                logger.debug(
                    f"Extracting beliefs from chunk {chunk_idx + 1}/{len(chunks)}"
                )
            chunk_beliefs = await self._extract_from_chunk(chunk_text, turn, source)
            all_beliefs.extend(chunk_beliefs)

        return all_beliefs

    async def _extract_from_chunk(
        self, chunk_text: str, turn: int, source: str
    ) -> List[Belief]:
        if self.config.extract_prompt_template != DEFAULT_EXTRACT_PROMPT:
            prompt_template = self.config.extract_prompt_template
        elif source == "user":
            prompt_template = self.config.extract_user_prompt_template
        else:
            prompt_template = self.config.extract_assistant_prompt_template

        # Support both {response} and {conversation} template variables
        if "{conversation}" in prompt_template:
            prompt = prompt_template.format(conversation=chunk_text)
        else:
            prompt = prompt_template.format(response=chunk_text)

        call = LLMCall(messages=[{"role": "user", "content": prompt}])

        BeliefListSchema = ExtractedBeliefSchema.build()

        try:
            llm_resp = await self.adapter.generate(
                call, response_format=BeliefListSchema
            )
            text = llm_resp.text.strip()
            data = recover_json_from_response(text)

            if data is None:
                return []

            if isinstance(data, dict):
                data = data.get("beliefs", data.get("root", []))
                if not isinstance(data, list):
                    data = []

            beliefs = []
            temp_beliefs = []

            for item in data:
                try:
                    subj = item.get("subject", "").strip()
                    subj_upper = subj.upper()
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

                    source_quote = item.get("source_quote", "")
                    if source_quote and len(source_quote) > 100:
                        source_quote = source_quote[:100]

                    b = Belief(
                        subject=subj,
                        predicate=item.get("predicate", ""),
                        value=normalize_value_format(item.get("value", "")),
                        confidence=float(item.get("confidence", 1.0)),
                        turn=turn,
                        source=source,
                        source_quote=source_quote,
                        category=item.get("category", ""),
                        belief_type=item.get("belief_type", "assertion"),
                        is_hypothetical=item.get("is_hypothetical", False),
                    )
                    temp_beliefs.append(b)
                except Exception as ve:
                    logger.warning(f"Skipping malformed belief: {ve}")

            if temp_beliefs:
                try:
                    texts_to_embed = [
                        f"{b.subject} {b.predicate} {b.value}" for b in temp_beliefs
                    ]
                    embeddings = await self.embedding_adapter.get_embeddings(
                        texts_to_embed
                    )
                    for b, emb in zip(temp_beliefs, embeddings):
                        b.embedding = emb
                        b.embedding_model = self.embedding_model
                        b.embedding_dim = self.embedding_dim
                        beliefs.append(b)
                except Exception as e:
                    logger.warning(
                        f"Error enqueuing batch embeddings: {e}. "
                        f"Falling back to individual embedding generation."
                    )
                    for b in temp_beliefs:
                        try:
                            text_to_embed = f"{b.subject} {b.predicate} {b.value}"
                            b.embedding = await self.embedding_adapter.get_embedding(
                                text_to_embed
                            )
                            b.embedding_model = self.embedding_model
                            b.embedding_dim = self.embedding_dim
                            beliefs.append(b)
                        except Exception as ie:
                            logger.error(
                                f"Error embedding individual belief fallback: {ie}",
                                exc_info=True,
                            )

            return beliefs
        except Exception as e:
            logger.error(f"Belief extraction error: {e}", exc_info=True)
            return []
