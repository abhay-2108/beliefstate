from typing import Any, List, Protocol, runtime_checkable, Optional
from beliefstate.call import LLMCall, LLMResponse

@runtime_checkable
class ProviderAdapter(Protocol):
    """Protocol for translating between native SDK formats and our universal models."""
    
    def to_llm_call(self, *args, **kwargs) -> LLMCall:
        """Convert native args/kwargs into a universal LLMCall."""
        ...
        
    def to_llm_response(self, response: Any) -> LLMResponse:
        """Convert a native SDK response object into a universal LLMResponse."""
        ...
        
    async def generate(self, call: LLMCall, response_format: Optional[Any] = None) -> LLMResponse:
        """Execute a generation request using this provider natively (used for internal tracker logic)."""
        ...
        
    async def get_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the text using this provider natively."""
        ...
        
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using this provider natively."""
        ...