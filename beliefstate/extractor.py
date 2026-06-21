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
    """Normalize numbers: remove commas/spaces, keep digits only."""
    # Replace comma-separated numbers with compact form: 5,000 -> 5000
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    return text


def normalize_currency(text: str) -> str:
    """Normalize currency to ISO code format: USD 5000 instead of $5,000."""
    # Common currency patterns
    patterns = [
        (r"\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", r"USD \1"),
        (r"€\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", r"EUR \1"),
        (r"£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", r"GBP \1"),
        (r"¥\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", r"JPY \1"),
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    # Handle currency: "five thousand dollars" normalized to digits
    # Best handled by LLM prompt, but add fallback
    text = re.sub(r"(USD|EUR|GBP|JPY)\s+(\d+),(\d+)", r"\1 \2\3", text)

    return text


def normalize_dates(text: str) -> str:
    """Normalize dates to ISO 8601 format (YYYY-MM-DD)."""
    # Month name to number mapping
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

    # Pattern: "March 15, 2024" -> 2024-03-15
    def replace_date_long(match: re.Match[str]) -> str:
        month_name = match.group(1).lower()
        day = match.group(2).zfill(2)
        year = match.group(3)
        month = months.get(month_name, "01")
        return f"{year}-{month}-{day}"

    text = re.sub(
        r"(" + "|".join(months.keys()) + r")\s+(\d{1,2}),?\s+(\d{4})",
        replace_date_long,
        text,
        flags=re.IGNORECASE,
    )

    # Pattern: "15 March 2024" -> 2024-03-15
    def replace_date_dmy(match: re.Match[str]) -> str:
        day = match.group(1).zfill(2)
        month_name = match.group(2).lower()
        year = match.group(3)
        month = months.get(month_name, "01")
        return f"{year}-{month}-{day}"

    text = re.sub(
        r"(\d{1,2})\s+(" + "|".join(months.keys()) + r")\s+(\d{4})",
        replace_date_dmy,
        text,
        flags=re.IGNORECASE,
    )

    # Pattern: MM/DD/YYYY (US standard)
    text = re.sub(
        r"(\d{1,2})/(\d{1,2})/(\d{4})",
        lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}",
        text,
    )

    return text


def normalize_percentages(text: str) -> str:
    """Normalize percentages to decimal format: 0.15 instead of 15%."""

    # Pattern: "15%" -> "0.15"
    def replace_percent(match: re.Match[str]) -> str:
        value = float(match.group(1)) / 100
        return f"{value:.2f}".rstrip("0").rstrip(".")

    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", replace_percent, text)
    return text


def normalize_value(value: str) -> str:
    """Apply all normalization rules to a belief value."""
    if not value:
        return value

    # Apply normalizations in order
    value = normalize_numbers(value)
    value = normalize_currency(value)
    value = normalize_dates(value)
    value = normalize_percentages(value)

    return value


def classify_response_type(text: str) -> str:
    """Classify the type of LLM response to determine extraction strategy.

    Response types:
    - "conversational": Natural language (extract beliefs)
    - "code": Code block (skip extraction)
    - "json": Pure JSON/structured data (skip extraction)
    - "sql": SQL query or output (skip extraction)
    - "markdown_heavy": Markdown with code (extract text only, skip code)
    """
    if not text:
        return "conversational"

    text = text.strip()

    # Check for JSON (array or object)
    if text.startswith("[") and text.endswith("]"):
        return "json"
    if text.startswith("{") and text.endswith("}"):
        try:
            json.loads(text)
            return "json"
        except json.JSONDecodeError:
            pass

    # Check for SQL
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

    # Check for code blocks
    code_pattern = r"```[\w]*\n"
    if re.search(code_pattern, text):
        # If mostly code blocks, classify as code
        code_blocks = re.findall(r"```[\w]*\n.*?\n```", text, re.DOTALL)
        total_code_chars = sum(len(block) for block in code_blocks)
        if total_code_chars / len(text) > 0.5:  # More than 50% is code
            return "code"
        else:
            return "markdown_heavy"

    # Check for inline code (single backticks)
    if text.count("`") > 4:  # Multiple code snippets
        return "markdown_heavy"

    return "conversational"


def chunk_response_by_paragraphs(text: str, max_chunk_length: int = 2000) -> List[str]:
    """Chunk a response into paragraphs for extraction.

    For long responses (>2000 chars), splits at paragraph boundaries
    (double newlines) to avoid token limits and improve extraction quality.
    """
    if not text or len(text) <= max_chunk_length:
        return [text]

    # Split by double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n\s*\n", text)

    if not paragraphs:
        return [text]

    # Group paragraphs into chunks respecting max_chunk_length
    chunks = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_length = len(para) + 2  # +2 for newlines

        # If adding this paragraph would exceed max length, save current chunk
        if current_chunk and current_length + para_length > max_chunk_length:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length

    # Add remaining chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    if not chunks:
        return [text]

    logger.debug(f"Chunked response into {len(chunks)} paragraphs for extraction")
    return chunks


def recover_json_from_response(text: str) -> Optional[List[Any]]:
    """Multi-layer JSON recovery for malformed LLM responses.

    Implements progressive recovery strategies:
    1. Direct JSON parse
    2. Remove markdown code blocks
    3. Extract JSON array from text
    4. Handle escaped quotes and unicode
    5. Attempt truncated JSON recovery (add closing ])
    6. Try to recover from incomplete objects
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # Layer 1: Direct parse
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

    # Layer 2: Remove markdown code blocks
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

    # Layer 3: Extract JSON array from surrounding text
    start_idx = text_clean.find("[")
    if start_idx == -1:
        return None

    # Search for matching closing bracket
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

    # Layer 4: Handle escaped quotes and unicode
    json_substr = json_substr.replace('"', '"').replace('"', '"')
    json_substr = json_substr.replace("'", '"')

    try:
        data = json.loads(json_substr)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Layer 5: Attempt truncated JSON recovery
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

    # Layer 6: Try to recover from incomplete object
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


class BeliefExtractor:
    def __init__(self, adapter: ProviderAdapter, config: TrackerConfig):
        self.adapter = adapter
        self.config = config
        # Try to extract model name and dimensionality from adapter
        self.embedding_model = getattr(adapter, "embed_model", "")
        self.embedding_dim = self._get_embedding_dim()

    def _get_embedding_dim(self) -> int:
        """Infer embedding dimensionality from model name."""
        model_name = self.embedding_model.lower()

        # Common embedding model dimensions
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

        # Fallback: query adapter if it supports dimension discovery
        if hasattr(self.adapter, "embedding_dim"):
            return int(self.adapter.embedding_dim)

        return 0  # Unknown dimensionality

    async def extract(
        self, response_text: str, turn: int, source: str = "assistant"
    ) -> List[Belief]:
        """Extract factual claims from text using the LLM adapter.

        Implements smart extraction with:
        - Response type classification (skip code/JSON/SQL)
        - Chunking for long responses (split at paragraph boundaries)
        - Robust JSON recovery for malformed responses
        """
        # Classify response type to skip non-conversational content
        resp_type = classify_response_type(response_text)

        if resp_type in ["code", "json", "sql"]:
            logger.debug(f"Skipping extraction for {resp_type} response type")
            return []

        # For markdown_heavy responses, strip code blocks
        extraction_text = response_text
        if resp_type == "markdown_heavy":
            # Remove code blocks from extraction
            extraction_text = re.sub(
                r"```[\w]*\n.*?\n```", "", response_text, flags=re.DOTALL
            )
            extraction_text = extraction_text.strip()
            if not extraction_text:
                logger.debug("Skipping extraction: only code blocks in response")
                return []

        # Chunk long responses at paragraph boundaries
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
        """Extract beliefs from a single chunk of text."""
        if self.config.extract_prompt_template != DEFAULT_EXTRACT_PROMPT:
            prompt_template = self.config.extract_prompt_template
        elif source == "user":
            prompt_template = self.config.extract_user_prompt_template
        else:
            prompt_template = self.config.extract_assistant_prompt_template

        prompt = prompt_template.format(response=chunk_text)

        # We make an internal LLM call to extract beliefs
        call = LLMCall(messages=[{"role": "user", "content": prompt}])

        from pydantic import RootModel

        class BeliefListSchema(RootModel[List[Belief]]):
            pass

        try:
            llm_resp = await self.adapter.generate(
                call, response_format=BeliefListSchema
            )

            # Parse JSON robustly using multi-layer recovery
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
                        value=normalize_value(item.get("value", "")),
                        confidence=float(item.get("confidence", 1.0)),
                        turn=turn,
                        source=source,
                    )
                    temp_beliefs.append(b)
                except ValidationError as ve:
                    logger.warning(f"Skipping malformed belief: {ve}")

            if temp_beliefs:
                try:
                    # Request batch embeddings for all valid beliefs
                    texts_to_embed = [
                        f"{b.subject} {b.predicate} {b.value}" for b in temp_beliefs
                    ]
                    embeddings = await self.adapter.get_embeddings(texts_to_embed)
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
                    # Fallback to individual embedding requests
                    for b in temp_beliefs:
                        try:
                            text_to_embed = f"{b.subject} {b.predicate} {b.value}"
                            b.embedding = await self.adapter.get_embedding(
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
            # Silently fail or log for extraction errors so we don't crash
            logger.error(f"Belief extraction error: {e}", exc_info=True)
            return []
