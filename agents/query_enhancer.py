"""
Query Enhancement Tool for Pharmaceutical Supply Chain Management System.
Improves query quality and context before routing to specialized agents.
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import openai
from loguru import logger
from pydantic import BaseModel

class QueryContext(BaseModel):
    """Structure for enhanced query context"""
    original_query: str
    enhanced_query: str
    domain_context: List[str]
    query_intent: str
    suggested_visualizations: List[str]
    metadata: Dict[str, Any]

class QueryEnhancer:
    """
    Enhances user queries by adding context, clarifying intent,
    and suggesting appropriate visualizations for pharmaceutical supply chain queries.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the query enhancer with OpenAI integration."""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        self._init_enhancement_patterns()
        logger.info("Query enhancer initialized successfully")
    
    def _init_enhancement_patterns(self):
        """Initialize patterns for query enhancement."""
        self.domain_contexts = {
            "inventory": [
                "pharmaceutical stock management",
                "medication inventory tracking",
                "supply chain optimization",
                "expiration date monitoring",
                "storage condition requirements"
            ],
            "transport": [
                "cold chain logistics",
                "shipment tracking",
                "delivery route optimization",
                "temperature monitoring",
                "transportation compliance"
            ],
            "compliance": [
                "regulatory requirements",
                "quality control standards",
                "documentation protocols",
                "safety guidelines",
                "audit procedures"
            ]
        }
        
        self.visualization_types = {
            "inventory": [
                "stock level charts",
                "expiration date timelines",
                "storage capacity utilization",
                "inventory turnover graphs",
                "reorder point indicators"
            ],
            "transport": [
                "shipment tracking maps",
                "delivery timeline charts",
                "temperature monitoring graphs",
                "route optimization maps",
                "carrier performance dashboards"
            ],
            "analytics": [
                "trend analysis charts",
                "comparative analysis graphs",
                "forecast projections",
                "risk assessment matrices",
                "cost analysis breakdowns"
            ]
        }
    
    def enhance_query(self, query: str) -> QueryContext:
        """
        Enhance the user's query by adding context and suggesting visualizations.
        
        Args:
            query: The original user query
            
        Returns:
            QueryContext object containing enhanced query and metadata
        """
        try:
            # Create enhancement prompt
            enhancement_prompt = self._create_enhancement_prompt(query)
            
            # Get enhanced query from OpenAI
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": """You are a pharmaceutical supply chain query enhancement specialist. 
                        Your task is to improve queries by adding relevant context and suggesting appropriate visualizations.
                        Focus on clarity, specificity, and actionable insights."""},
                        {"role": "user", "content": enhancement_prompt}
                    ],
                    temperature=0.2
                )
                
                # Parse the enhanced response
                enhanced_data = self._parse_enhancement_response(response.choices[0].message.content)
            except Exception as api_error:
                logger.error(f"OpenAI API error during query enhancement: {str(api_error)}")
                # Fall back to a simplified enhancement using the original query
                logger.info(f"Falling back to original query: {query}")
                return self._generate_fallback_context(query, f"API Error: {str(api_error)}")
            
            # Create QueryContext
            query_context = QueryContext(
                original_query=query,
                enhanced_query=enhanced_data["enhanced_query"],
                domain_context=enhanced_data["domain_context"],
                query_intent=enhanced_data["query_intent"],
                suggested_visualizations=enhanced_data["visualizations"],
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "confidence_score": enhanced_data["confidence"],
                    "enhancement_type": enhanced_data["enhancement_type"]
                }
            )
            
            logger.info(f"Query enhanced successfully: {query} -> {query_context.enhanced_query}")
            return query_context
            
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            return self._generate_error_context(query, str(e))
    
    def _create_enhancement_prompt(self, query: str) -> str:
        """Create a prompt for query enhancement."""
        return f"""Please enhance the following pharmaceutical supply chain query:

Query: {query}

Please provide:
1. Enhanced version of the query with more specific context
2. Relevant domain context (from: {', '.join(self.domain_contexts.keys())})
3. Clear query intent
4. Suggested visualizations
5. Confidence score (0-1)
6. Enhancement type (clarification/expansion/specification)

Format the response as:
ENHANCED_QUERY: <enhanced query>
DOMAIN_CONTEXT: <list of relevant contexts>
QUERY_INTENT: <clear intent>
VISUALIZATIONS: <list of suggested visualizations>
CONFIDENCE: <score>
ENHANCEMENT_TYPE: <type>"""
    
    def _parse_enhancement_response(self, response: str) -> Dict[str, Any]:
        """Parse the enhancement response from OpenAI."""
        lines = response.strip().split('\n')
        enhanced_data = {
            "enhanced_query": "",
            "domain_context": [],
            "query_intent": "",
            "visualizations": [],
            "confidence": 0.0,
            "enhancement_type": ""
        }
        
        for line in lines:
            if line.startswith("ENHANCED_QUERY:"):
                enhanced_data["enhanced_query"] = line.replace("ENHANCED_QUERY:", "").strip()
            elif line.startswith("DOMAIN_CONTEXT:"):
                contexts = line.replace("DOMAIN_CONTEXT:", "").strip()
                enhanced_data["domain_context"] = [c.strip() for c in contexts.split(',')]
            elif line.startswith("QUERY_INTENT:"):
                enhanced_data["query_intent"] = line.replace("QUERY_INTENT:", "").strip()
            elif line.startswith("VISUALIZATIONS:"):
                visuals = line.replace("VISUALIZATIONS:", "").strip()
                enhanced_data["visualizations"] = [v.strip() for v in visuals.split(',')]
            elif line.startswith("CONFIDENCE:"):
                enhanced_data["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
            elif line.startswith("ENHANCEMENT_TYPE:"):
                enhanced_data["enhancement_type"] = line.replace("ENHANCEMENT_TYPE:", "").strip()
        
        return enhanced_data
    
    def _generate_error_context(self, query: str, error: str) -> QueryContext:
        """Generate error context when enhancement fails."""
        return QueryContext(
            original_query=query,
            enhanced_query=query,  # Return original query if enhancement fails
            domain_context=[],
            query_intent="error",
            suggested_visualizations=[],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "error": error,
                "confidence_score": 0.0,
                "enhancement_type": "error"
            }
        )
    
    def _generate_fallback_context(self, query: str, error_msg: str) -> QueryContext:
        """Generate fallback context when API enhancement fails but we can still proceed."""
        # For inventory queries, add basic inventory domain context
        domain_context = []
        if any(term in query.lower() for term in ["inventory", "stock", "supply", "level"]):
            domain_context = ["pharmaceutical stock management"]
        elif any(term in query.lower() for term in ["transport", "shipping", "delivery"]):
            domain_context = ["pharmaceutical transportation"]
        
        return QueryContext(
            original_query=query,
            enhanced_query=query,  # Use the original query
            domain_context=domain_context,
            query_intent="direct_query",
            suggested_visualizations=[],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "error": error_msg,
                "confidence_score": 0.7,  # Reasonable confidence in the original query
                "enhancement_type": "fallback"
            }
        )

# For direct testing
if __name__ == "__main__":
    enhancer = QueryEnhancer()
    
    # Test queries
    test_queries = [
        "Show me insulin stock",
        "Where are my shipments",
        "Check compliance",
        "List expired medicines",
        "Track cold chain"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        result = enhancer.enhance_query(query)
        print(f"Enhanced result: {result}") 