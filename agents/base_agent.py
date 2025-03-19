from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os

class AgentType(Enum):
    """Enum for different types of agents."""
    QUERY_ENHANCEMENT = "query_enhancement"
    RAG = "rag"
    WEB_SEARCH = "web_search"
    ENHANCED_ANALYTICS = "enhanced_analytics"
    FALLBACK = "fallback"
    CAG = "cag"
    ROUTER = "router"
    AGGREGATOR = "aggregator"

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    system_prompt: str = "You are a helpful assistant."
    tools: Optional[List[str]] = None

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.config = config or AgentConfig(
            model_name=model_name,
            temperature=0.7,
            max_tokens=1000
        )
    
    @abstractmethod
    async def process(self, query: str, **kwargs) -> str:
        """Process a query and return a response."""
        pass
    
    @abstractmethod
    def get_type(self) -> AgentType:
        """Get the type of this agent."""
        pass 