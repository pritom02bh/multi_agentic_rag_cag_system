"""
Router Agent with Cache Augmented Generation (CAG) for Pharmaceutical Supply Chain Management.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime
import time
from enum import Enum
import json
import traceback

# Define casual conversation keywords and responses
CASUAL_KEYWORDS = {
    'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
    'good evening', 'howdy', 'what\'s up', 'how are you', 'how\'s it going'
}

QUICK_RESPONSES = {
    'hi': 'Hello! How can I assist you with our pharmaceutical supply chain today?',
    'hello': 'Hi there! Ready to help you with pharmaceutical supply chain management.',
    'hey': 'Hey! How can I help you with supply chain operations today?',
    'how are you': 'I\'m functioning well and ready to assist with your pharmaceutical supply chain queries!',
    'good morning': 'Good morning! How can I help you with supply chain management today?',
    'good afternoon': 'Good afternoon! Ready to assist with your pharmaceutical supply chain needs.',
    'good evening': 'Good evening! How can I help you with supply chain operations?'
}

import openai
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel
import numpy as np
from loguru import logger

from .query_enhancer import QueryContext, QueryEnhancer
from .augmented_tools import output_processor
from .aggregator_agent import AggregatorAgent
from .base_agent import BaseAgent, AgentType
from .cag_service import CAGService

class QueryType(str, Enum):
    """Query types for routing decisions"""
    INVENTORY = "inventory"
    TRANSPORT = "transport"
    POLICY = "policy"
    GUIDELINES = "guidelines"
    ANALYTICS = "analytics"
    REAL_TIME = "real_time"
    COMPLIANCE = "compliance"
    RISK = "risk"
    COST = "cost"
    VISUALIZATION = "visualization"
    FORECAST = "forecast"
    GENERAL = "general"
    ERROR = "error"
    NEWS = "news"

class AnalysisType(str, Enum):
    """Types of analysis that can be performed"""
    TREND = "trend"
    COMPARISON = "comparison"
    FORECAST = "forecast"
    CORRELATION = "correlation"
    RISK = "risk"
    OPTIMIZATION = "optimization"
    IMPACT = "impact"

class CachedKnowledge(BaseModel):
    """Structure for cached knowledge"""
    namespace: str
    content: str
    metadata: Dict[str, Any]
    last_updated: datetime

class RouterAgent(BaseAgent):
    """
    Router agent that directs queries to appropriate specialized agents based on content analysis.
    Implements a hybrid architecture using RAG for inventory data and CAG for policies and transport.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize query enhancer
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.query_enhancer = QueryEnhancer(openai_api_key=self.openai_api_key)
        
        # Initialize CAG service for context retrieval
        self.cag_service = CAGService(force_use_openai=True)
        self.cag_service.preload_knowledge("data")
        
        # Initialize aggregator
        self.aggregator = AggregatorAgent(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            max_tokens=1500,
            openai_api_key=self.openai_api_key
        )
        
        # Initialize cached knowledge
        self.knowledge_cache = {qt: {} for qt in QueryType}
        
        # Initialize query patterns
        self._init_query_patterns()
        
        logger.info("Enhanced router agent initialized with hybrid RAG/CAG architecture")
    
    def _init_query_patterns(self):
        """Initialize comprehensive query classification patterns."""
        self.patterns = {
            QueryType.INVENTORY: {
                "keywords": {
                    # Basic inventory terms
                    "inventory", "stock", "supply", "quantity", "level",
                    "storage", "warehouse", "item", "product", "medicine",
                    # Specific metrics
                    "reorder point", "maximum level", "minimum level", "safety stock",
                    "turnover", "stockout", "overstock", "expiry", "expired",
                    # Actions
                    "replenish", "restock", "order", "receive", "count",
                    # Specific items
                    "insulin", "paracetamol", "ibuprofen", "antibiotic",
                    # Time-related
                    "current", "historical", "trend", "forecast"
                },
                "analysis_types": {
                    AnalysisType.TREND,
                    AnalysisType.FORECAST,
                    AnalysisType.OPTIMIZATION
                },
                "namespaces": ["inventory"],
                "primary_agent": AgentType.RAG,
                "secondary_agents": [AgentType.ENHANCED_ANALYTICS],
                "visualization": True
            },
            QueryType.TRANSPORT: {
                "keywords": {
                    # Transport modes
                    "transport", "shipping", "shipment", "delivery", "carrier",
                    "air", "road", "route", "freight", "cargo",
                    # Tracking
                    "track", "tracking", "location", "status", "eta",
                    "arrival", "departure", "delay", "schedule",
                    # Conditions
                    "temperature", "humidity", "pressure", "excursion",
                    # Documentation
                    "customs", "clearance", "documentation", "paperwork",
                    # Quality
                    "integrity", "damage", "compliance", "violation",
                    # History
                    "history", "past", "record", "log"
                },
                "analysis_types": {
                    AnalysisType.TREND,
                    AnalysisType.IMPACT,
                    AnalysisType.RISK
                },
                "namespaces": ["transport"],
                "primary_agent": AgentType.CAG,
                "secondary_agents": [],
                "visualization": True
            },
            QueryType.POLICY: {
                "keywords": {
                    # Policy terms
                    "policy", "policies", "guideline", "guidelines", "rule",
                    "regulation", "standard", "procedure", "protocol",
                    # Actions
                    "manage", "implement", "follow", "adhere", "maintain",
                    # Specific policies
                    "inventory management", "safety stock", "reorder point",
                    "ABC analysis", "turnover", "holding cost", "obsolescence",
                    # Business terms
                    "service level", "categorization", "replenishment"
                },
                "analysis_types": {
                    AnalysisType.IMPACT
                },
                "namespaces": ["policy"],
                "primary_agent": AgentType.CAG,
                "secondary_agents": [],
                "visualization": False
            },
            QueryType.GUIDELINES: {
                "keywords": {
                    # Guideline terms
                    "guideline", "regulation", "requirement", "compliance",
                    "standard", "fda", "phmsa", "tsa", "regulatory",
                    # Specific guidelines
                    "cold chain", "temperature control", "sanitary", "storage",
                    "transportation", "handling", "documentation",
                    # Government
                    "government", "federal", "authority", "agency",
                    # Compliance
                    "21 cfr", "fsma", "fsma sanitary", "part 1", "subpart o"
                },
                "analysis_types": {
                    AnalysisType.RISK,
                    AnalysisType.IMPACT
                },
                "namespaces": ["guidelines"],
                "primary_agent": AgentType.CAG,
                "secondary_agents": [],
                "visualization": False
            }
        }
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the enhanced workflow:
        1. Enhance the query using QueryEnhancer
        2. Identify query type
        3. Route to appropriate agents (RAG, CAG, WebSearch, or Analytics)
        4. Aggregate responses
        5. Format for UI
        """
        # Clear previous outputs
        output_processor.clear()
        
        try:
            # Step 1: Enhance the query
            try:
                enhanced_query_context = self.query_enhancer.enhance_query(query)
                enhanced_query = enhanced_query_context.enhanced_query
                
                # Check if enhancement failed and log appropriately
                if enhanced_query_context.query_intent == "error":
                    logger.warning(f"Query enhancement failed, using original query: {query}")
                    if 'error' in enhanced_query_context.metadata:
                        logger.error(f"Enhancement error: {enhanced_query_context.metadata['error']}")
                    
                    # For errors that indicate API issues, use original query but add warning
                    if 'api' in str(enhanced_query_context.metadata.get('error', '')).lower():
                        logger.warning("Possible API connection issue detected")
                
                logger.info(f"Enhanced query: {enhanced_query}")
            except Exception as e:
                logger.error(f"Error during query enhancement: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Fall back to original query if enhancement fails
                enhanced_query = query
                logger.warning(f"Using original query due to enhancement failure: {query}")
            
            # Step 2: Identify query type
            query_type = self._identify_query_type(enhanced_query)
            confidence = self._calculate_confidence(enhanced_query, query_type)
            
            # If confidence is too low, generate error response
            if confidence < 0.5:
                return self._generate_error_response("Low confidence in query classification")
            
            # Step 3: Route to appropriate agents based on query type
            agent_responses = {}
            
            # For inventory queries, use RAG agent
            if query_type == QueryType.INVENTORY:
                logger.info(f"Routing inventory query to RAG agent: {enhanced_query}")
                rag_response = await self._process_with_rag(enhanced_query, ["inventory"])
                agent_responses["rag"] = rag_response
            
            # For transport queries, use CAG
            elif query_type == QueryType.TRANSPORT:
                logger.info(f"Routing transport query to CAG: {enhanced_query}")
                cag_response = self.cag_service.query(enhanced_query, "transport")
                agent_responses["cag"] = cag_response
            
            # For policy queries, use CAG
            elif query_type == QueryType.POLICY:
                logger.info(f"Routing policy query to CAG: {enhanced_query}")
                cag_response = self.cag_service.query(enhanced_query, "policy")
                agent_responses["cag"] = cag_response
            
            # For guidelines queries, use CAG
            elif query_type == QueryType.GUIDELINES:
                logger.info(f"Routing guidelines query to CAG: {enhanced_query}")
                cag_response = self.cag_service.query(enhanced_query, "guidelines")
                agent_responses["cag"] = cag_response
                
            # For real-time or news queries, use WebSearchAgent
            elif query_type == QueryType.REAL_TIME or query_type == QueryType.NEWS:
                logger.info(f"Routing real-time query to WebSearchAgent: {enhanced_query}")
                web_response = await self._process_with_web_search(enhanced_query)
                agent_responses["web_search"] = web_response
                
            # For analytics or visualization queries, use Analytics agent
            elif query_type == QueryType.ANALYTICS or query_type == QueryType.VISUALIZATION:
                logger.info(f"Routing analytics query to Analytics agent: {enhanced_query}")
                analytics_response = await self._process_with_analytics(enhanced_query)
                agent_responses["analytics"] = analytics_response
            
            # For other queries, use both RAG and CAG
            else:
                logger.info(f"Routing general query to multiple agents: {enhanced_query}")
                # Get RAG response for inventory
                rag_response = await self._process_with_rag(enhanced_query, ["inventory"])
                agent_responses["rag"] = rag_response
                
                # Get CAG response for policies and guidelines
                cag_response = self.cag_service.query(enhanced_query)
                agent_responses["cag"] = cag_response
                
                # For comprehensive queries, also try web search if appropriate
                if any(term in enhanced_query.lower() for term in ["recent", "latest", "news", "trends"]):
                    web_response = await self._process_with_web_search(enhanced_query)
                    agent_responses["web_search"] = web_response
            
            # Step 4: Aggregate responses
            logger.info(f"Agent responses: {agent_responses}")
            
            # Skip aggregation if we only have one response
            if len(agent_responses) == 1:
                # Get the first (and only) response
                agent_type, response_data = next(iter(agent_responses.items()))
                
                # Ensure it has required fields
                if isinstance(response_data, dict):
                    if 'response' not in response_data and 'content' in response_data:
                        response_data['response'] = response_data['content']
                        
                    # Add metadata
                    if 'metadata' not in response_data:
                        response_data['metadata'] = {}
                    
                    response_data['metadata']['query_type'] = query_type.value
                    response_data['metadata']['confidence'] = confidence
                    response_data['query'] = enhanced_query
                    
                    # No need to format, return directly
                    return response_data
            
            # Otherwise, aggregate multiple responses  
            aggregated_response = await self.aggregator.process_outputs(
                agent_responses, 
                enhanced_query, 
                {"query_type": query_type.value, "confidence": confidence}
            )
            
            # Step 5: Format for UI
            logger.info(f"Aggregated response keys: {aggregated_response.keys()}")
            
            # Skip format_for_ui and return directly to let app.py handle response formatting
            return aggregated_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self._generate_error_response(f"Error: {str(e)}")
    
    async def _process_with_rag(self, query: str, namespaces: List[str]) -> Dict[str, Any]:
        """Process a query using the RAG agent."""
        try:
            from .agent_registry import get_agent_registry
            registry = get_agent_registry()
            rag_agent = registry.get_agent(AgentType.RAG)
            
            if not rag_agent:
                return {"response": "RAG agent not available", "error": True}
            
            response = await rag_agent.process(query)
            
            # Handle different response formats
            if isinstance(response, str):
                return {"response": response, "source_type": "rag"}
            elif isinstance(response, dict):
                # Ensure response and content fields are present
                if "response" not in response:
                    response["response"] = response.get("content", "No content available")
                
                # For inventory queries, we want to ensure the context is included
                # to allow for more detailed formatting in the UI
                if any(word in query.lower() for word in ["inventory", "stock", "level", "item", "product", "list"]):
                    if "context" not in response and hasattr(rag_agent, "last_context"):
                        response["context"] = rag_agent.last_context
                
                response["source_type"] = "rag"
                return response
            else:
                return {"response": f"Unexpected response type: {type(response)}", "error": True}
                
        except Exception as e:
            logger.error(f"Error processing with RAG: {str(e)}")
            return {"response": f"Error with RAG processing: {str(e)}", "error": True}
    
    async def _process_with_web_search(self, query: str) -> Dict[str, Any]:
        """Process a query using the WebSearchAgent."""
        try:
            from .agent_registry import get_agent_registry
            registry = get_agent_registry()
            web_agent = registry.get_agent(AgentType.WEB_SEARCH)
            
            if not web_agent:
                return {"response": "Web search agent not available", "error": True}
            
            response = await web_agent.process(query)
            
            # Handle different response formats
            if isinstance(response, str):
                return {"response": response, "source_type": "web_search"}
            elif isinstance(response, dict):
                if "response" not in response:
                    response["response"] = response.get("content", "No content available")
                response["source_type"] = "web_search"
                return response
            else:
                return {"response": f"Unexpected response type: {type(response)}", "error": True}
        except Exception as e:
            logger.error(f"Error processing with web search: {str(e)}")
            return {"response": f"Error with web search: {str(e)}", "error": True}
    
    async def _process_with_analytics(self, query: str) -> Dict[str, Any]:
        """Process a query using the Analytics agent."""
        try:
            from .agent_registry import get_agent_registry
            registry = get_agent_registry()
            analytics_agent = registry.get_agent(AgentType.ENHANCED_ANALYTICS)
            
            if not analytics_agent:
                return {"response": "Analytics agent not available", "error": True}
            
            response = await analytics_agent.process(query)
            
            # Handle different response formats
            if isinstance(response, str):
                return {"response": response, "source_type": "analytics"}
            elif isinstance(response, dict):
                if "response" not in response:
                    response["response"] = response.get("content", "No content available")
                response["source_type"] = "analytics"
                return response
            else:
                return {"response": f"Unexpected response type: {type(response)}", "error": True}
        except Exception as e:
            logger.error(f"Error processing with analytics: {str(e)}")
            return {"response": f"Error with analytics: {str(e)}", "error": True}
    
    def _identify_query_type(self, query: str) -> QueryType:
        """Identify the type of query based on keywords and patterns."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Check for presence of keywords for each query type
        type_scores = {}
        
        for query_type, pattern in self.patterns.items():
            keywords = pattern.get("keywords", set())
            
            # Calculate score based on keyword matches
            matched_keywords = keywords.intersection(query_words)
            
            # Also check for phrase matches
            phrase_score = 0
            for keyword in keywords:
                if " " in keyword and keyword in query_lower:
                    phrase_score += 1
                    matched_keywords.add(keyword)
            
            # Calculate score
            match_score = len(matched_keywords) + phrase_score * 2  # Phrases count double
            
            if match_score > 0:
                # Apply additional boosts based on query characteristics
                temporal_boost = self._calculate_temporal_boost(query_words)
                complexity_boost = self._calculate_complexity_boost(query_words)
                
                # Calculate final score
                type_scores[query_type] = match_score + temporal_boost + complexity_boost
        
        # Get query type with highest score, or default to general
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return QueryType.GENERAL
    
    def _calculate_temporal_boost(self, query_words: Set[str]) -> float:
        """Calculate boost for temporal queries."""
        temporal_words = {"current", "now", "latest", "today", "recent", "last", "previous", "future", "upcoming", "forecast"}
        temporal_matches = temporal_words.intersection(query_words)
        return len(temporal_matches) * 0.5  # Each temporal word adds 0.5 to score
    
    def _calculate_complexity_boost(self, query_words: Set[str]) -> float:
        """Calculate boost based on query complexity."""
        complexity_words = {"analyze", "analysis", "compare", "comparison", "trend", "trends", "pattern", "patterns", "correlation", "impact"}
        complexity_matches = complexity_words.intersection(query_words)
        return len(complexity_matches) * 0.7  # Each complexity word adds 0.7 to score
    
    def _calculate_confidence(self, query: str, query_type: QueryType) -> float:
        """Calculate confidence score for the query classification."""
        # Base confidence level
        base_confidence = self._get_base_confidence(query_type)
        
        # Pattern match confidence
        pattern_confidence = self._calculate_pattern_confidence(query, query_type)
        
        # Query length factor - longer queries typically have more context
        length_factor = min(len(query.split()) / 15, 1.0) * 0.1
        
        # Calculate final confidence score
        confidence = base_confidence + pattern_confidence + length_factor
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def _get_base_confidence(self, query_type: QueryType) -> float:
        """Get base confidence for query type."""
        base_confidence = {
            QueryType.INVENTORY: 0.7,
            QueryType.TRANSPORT: 0.7,
            QueryType.POLICY: 0.7,
            QueryType.GUIDELINES: 0.7,
            QueryType.ANALYTICS: 0.6,
            QueryType.GENERAL: 0.5,
            QueryType.ERROR: 0.3,
        }
        return base_confidence.get(query_type, 0.5)
    
    def _calculate_pattern_confidence(self, query: str, query_type: QueryType) -> float:
        """Calculate confidence based on pattern matching."""
        if query_type not in self.patterns:
            return 0.0
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Get pattern for query type
        pattern = self.patterns[query_type]
        keywords = pattern.get("keywords", set())
        
        # Calculate pattern match ratio
        matched_keywords = keywords.intersection(query_words)
        match_ratio = len(matched_keywords) / max(len(query_words), 1)
        
        # Also check for phrase matches
        phrase_matches = 0
        for keyword in keywords:
            if " " in keyword and keyword in query_lower:
                phrase_matches += 1
        
        phrase_factor = min(phrase_matches / 2, 1.0) * 0.2
        
        return match_ratio * 0.2 + phrase_factor
    
    def _generate_error_response(self, error: str) -> Dict[str, Any]:
        """Generate an error response."""
        return {
            "response": f"I'm sorry, but I encountered an error while processing your request: {error}",
            "query_type": QueryType.ERROR.value,
            "confidence": 0.0,
            "error": True,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "error": error
            }
        }
    
    def get_type(self) -> AgentType:
        """Get the type of this agent."""
        return AgentType.ROUTER
    
    async def process(self, query: str, **kwargs) -> str:
        """Process a query and return a response."""
        result = await self.process_query(query)
        return result.get("response", "Unable to process query")

# For direct testing
if __name__ == "__main__":
    router = RouterAgent()
    
    # Cache some test knowledge
    router.cache_knowledge(
        namespace="inventory",
        content="Current insulin stock levels are at 75% capacity.",
        metadata={"last_updated": "2025-03-18T12:00:00"}
    )
    
    # Test queries
    test_queries = [
        "What is the current stock level of insulin?",
        "Show me delayed shipments from last week",
        "What are our cold chain compliance requirements?",
        "List all medications expiring this month",
        "Give me the latest transport updates"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        result = router.process_query(query)
        print(f"Routing result: {result}")

async def main():
    # Initialize CAG service
    cag_service = CAGService(force_use_openai=True)
    cag_service.preload_knowledge("data")

    # Initialize agents
    router_agent = RouterAgent(config)
    analytics_agent = AnalyticsVisualizationAgent(config, cag_service)

    # Example queries
    queries = [
        "What are our current transport policies?",
        "Show me the guidelines for cold chain management",
        "Analyze our transport performance"
    ]

    for query in queries:
        result = await router_agent.route_query(query)
        print(f"Query: {query}")
        print(f"Response: {result['response']}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 