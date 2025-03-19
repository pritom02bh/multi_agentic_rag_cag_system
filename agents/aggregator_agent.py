import os
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
import json
import re
from langchain_core.output_parsers import StrOutputParser
from enum import Enum
from pydantic import BaseModel, Field
import pandas as pd
from tabulate import tabulate
import time
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .augmented_tools import AgentOutput, output_processor
from .base_agent import BaseAgent, AgentType
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SourceType(str, Enum):
    """Enum for source types with credibility weights"""
    INTERNAL_RAG = "rag"  # Weight: 1.0
    ANALYTICS = "analytics"  # Weight: 0.9
    WEB_NEWS = "web_news_search"  # Weight: 0.7
    UNKNOWN = "unknown"

class ConflictResolution(str, Enum):
    """Enum for conflict resolution strategies"""
    PREFER_INTERNAL = "prefer_internal"
    FLAG_CONFLICT = "flag_conflict"
    MERGE_ALL = "merge_all"
    MOST_RECENT = "most_recent"
    HIGHEST_CONFIDENCE = "highest_confidence"
    INTERNAL_PRIORITY = "internal_priority"

class ResponseSection(str, Enum):
    """Enum for response sections"""
    SUMMARY = "summary"
    DETAILS = "details"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    SOURCES = "sources"
    ALERTS = "alerts"
    RECOMMENDATIONS = "recommendations"
    REFERENCES = "references"

class EnhancedResponse(BaseModel):
    main_content: str = Field(description="The main textual content of the response")
    key_points: List[str] = Field(description="Key points extracted from the response")
    data_insights: Optional[Dict[str, Any]] = Field(description="Structured data insights if available")
    visualization_suggestions: List[str] = Field(description="Suggestions for visualizations")
    metadata: Dict[str, Any] = Field(description="Relevant metadata about the response")

class AggregatorAgent(BaseAgent):
    """
    Enhanced Aggregator agent for pharmaceutical supply chain management system.
    
    This agent combines and formats responses from multiple specialized agents
    into a coherent, well-structured final response.
    """
    
    def __init__(self, model=None, max_tokens=1500, openai_api_key=None):
        """Initialize the aggregator agent."""
        # No need to call super().__init__ since we're managing our own config
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.max_tokens = max_tokens
        
        try:
            # Initialize OpenAI client
            self.client = openai.OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
            logger.info(f"Initialized AggregatorAgent with model {self.model}")
        except Exception as e:
            logger.error(f"Error initializing AggregatorAgent: {str(e)}")
            raise
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=self.model,
            temperature=0.0,
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Source credibility weights
        self.source_weights = {
            SourceType.INTERNAL_RAG: 1.0,
            SourceType.ANALYTICS: 0.9,
            SourceType.WEB_NEWS: 0.7,
            SourceType.UNKNOWN: 0.4
        }
        
        # Create the aggregation prompt
        self.aggregation_prompt = self._create_aggregation_prompt()
        
        # Create the aggregation chain
        self.aggregation_chain = (
            RunnablePassthrough() 
            | self.aggregation_prompt 
            | self.llm
            | StrOutputParser()
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=EnhancedResponse)
        
        self.enhance_template = """You are an expert in processing and enhancing information.
        Your task is to create a well-structured, clear, and concise response from multiple sources.
        
        Guidelines:
        1. Remove redundant or unnecessary information
        2. Highlight key insights and findings
        3. Suggest relevant visualizations if data is present
        4. Structure the response in a clear format
        5. Include only the most relevant sources
        
        Input Information:
        {raw_content}
        
        Data Available:
        {data_content}
        
        Visualization Options:
        {viz_options}
        
        Please process this information and provide:
        1. A clear main content
        2. Key points (bullet points)
        3. Data insights (if applicable)
        4. Visualization suggestions
        5. Relevant metadata
        
        {format_instructions}
        """
        
        self.enhance_prompt = ChatPromptTemplate.from_template(
            template=self.enhance_template
        )
    
    def get_type(self) -> AgentType:
        """Get the type of this agent."""
        return AgentType.AGGREGATOR
    
    async def process(self, query: str, **kwargs) -> str:
        """Process a query and return a response."""
        # In the aggregator's case, we don't process single queries directly
        # This method is implemented to comply with the BaseAgent interface
        return f"The Aggregator agent doesn't process individual queries. Use process_outputs instead."

    def _create_aggregation_prompt(self) -> PromptTemplate:
        """Create an enhanced prompt template for aggregation."""
        template = """
        You are an expert in pharmaceutical supply chain management. Your task is to create a coherent and informative response by combining information from multiple specialized agents.

        Original query: {query}

        Agent responses to integrate:
        {agent_responses}

        Please create a comprehensive response that:
        1. Prioritizes internal data over external sources
        2. Highlights any conflicts between sources
        3. Integrates visualizations and analytics naturally
        4. Cites sources appropriately
        5. Maintains clear section organization

        Use this structure:
        ## Summary
        [Brief overview of key findings]

        ## Details
        [Main content with integrated information]

        ## Analysis
        [Interpretation and insights]

        ## Recommendations
        [Action items and next steps]

        If there are conflicts between sources, note them explicitly:
        - Internal Data: [information]
        - External Source: [conflicting information]
        - Resolution: [explanation of which to trust and why]

        Final response:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["query", "agent_responses"]
        )

    def process_responses(
        self,
        query: str,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process and combine responses from multiple agents."""
        try:
            # Validate and identify primary response
            primary_response = self._identify_primary_response(responses)
            if not primary_response:
                logger.error("No valid primary response found")
                return self._generate_empty_response(query)
            
            # Convert responses to proper format
            formatted_responses = []
            for response in responses:
                if isinstance(response, str):
                    try:
                        response = json.loads(response)
                    except json.JSONDecodeError:
                        response = {
                            "summary": response,
                            "source_type": "unknown"
                        }
                formatted_responses.append(response)
            
            # Extract key information
            summary_parts = []
            details_parts = []
            analysis_parts = []
            recommendations_parts = []
            visualization_data = None
            web_search_data = []
            rag_data = []
            
            # Process each response
            for response in formatted_responses:
                # Extract sections
                if summary := response.get("summary"):
                    summary_parts.append(summary)
                if details := response.get("details"):
                    details_parts.append(details)
                if analysis := response.get("analysis"):
                    analysis_parts.append(analysis)
                if recommendations := response.get("recommendations"):
                    recommendations_parts.append(recommendations)
                
                # Get visualization if available
                if viz := response.get("visualization"):
                    visualization_data = viz
                
                # Collect source data
                source_type = response.get("source_type", "unknown")
                if source_type == "web_news_search":
                    web_search_data.append(response)
                elif source_type == "rag":
                    rag_data.append(response)
            
            # For inventory queries, ensure proper formatting
            if any("inventory" in r.get("query", "").lower() for r in formatted_responses):
                # Get the RAG response with inventory data
                inventory_response = next(
                    (r for r in formatted_responses if r.get("source_type") == "rag"),
                    None
                )
                
                if inventory_response:
                    # Use the structured sections from the RAG response
                    sections = {
                        "summary": inventory_response.get("summary", ""),
                        "details": inventory_response.get("details", ""),
                        "analysis": inventory_response.get("analysis", ""),
                        "recommendations": inventory_response.get("recommendations", "")
                    }
                    
                    # Add web search context if available
                    if web_search_data:
                        web_context = self._format_web_context(web_search_data)
                        if web_context:
                            sections["analysis"] += f"\n\n### Market Context\n{web_context}"
                    
                    # Create final response
                    return {
                        "type": "aggregated",
                        "query": query,
                        "response": "\n\n".join(section for section in sections.values() if section),
                        "sections": sections,
                        "visualization": visualization_data or inventory_response.get("visualization"),
                        "source_type": "aggregated",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "sources": [r.get("source_type", "unknown") for r in formatted_responses],
                            "confidence": self._calculate_confidence(formatted_responses),
                            "processing_time": sum(float(r.get("processing_time", 0)) for r in formatted_responses),
                            "result_count": sum(int(r.get("result_count", 0)) for r in formatted_responses)
                        }
                    }
            
            # For other queries, use standard formatting
            sections = {
                "summary": self._format_section(summary_parts, "Summary"),
                "details": self._format_section(details_parts, "Details"),
                "analysis": self._format_section(analysis_parts, "Analysis"),
                "recommendations": self._format_section(recommendations_parts, "Recommendations")
            }
            
            # Create final response
            return {
                "type": "aggregated",
                "query": query,
                "response": "\n\n".join(section for section in sections.values() if section),
                "sections": sections,
                "visualization": visualization_data,
                "source_type": "aggregated",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "sources": [r.get("source_type", "unknown") for r in formatted_responses],
                    "confidence": self._calculate_confidence(formatted_responses),
                    "processing_time": sum(float(r.get("processing_time", 0)) for r in formatted_responses),
                    "result_count": sum(int(r.get("result_count", 0)) for r in formatted_responses)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing responses: {str(e)}")
            return self._generate_error_response(query, str(e))
    
    def _format_section(self, parts: List[str], section_name: str) -> str:
        """Format a section with proper headers and content."""
        if not parts:
            return ""
        
        # Remove duplicate content
        unique_parts = []
        seen = set()
        for part in parts:
            if part not in seen:
                unique_parts.append(part)
                seen.add(part)
        
        # Format section
        formatted_parts = []
        for part in unique_parts:
            # Remove existing headers if present
            part = re.sub(r'^#+\s*' + section_name + r'\s*\n', '', part, flags=re.MULTILINE)
            formatted_parts.append(part.strip())
        
        return f"## {section_name}\n" + "\n\n".join(formatted_parts)
    
    def _calculate_confidence(self, responses: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score from responses."""
        if not responses:
            return 0.0
        
        confidences = [float(r.get("confidence", 0)) for r in responses]
        if not confidences:
            return 0.0
        
        # Weight confidences by source type
        weighted_confidences = []
        for response, confidence in zip(responses, confidences):
            source_type = response.get("source_type", "unknown")
            weight = self.source_weights.get(source_type, self.source_weights[SourceType.UNKNOWN])
            weighted_confidences.append(confidence * weight)
        
        return sum(weighted_confidences) / len(weighted_confidences)
    
    def _generate_empty_response(self, query: str) -> Dict[str, Any]:
        """Generate an empty response structure."""
        return {
            "type": "aggregated",
            "query": query,
            "response": "I apologize, but I couldn't find any relevant information for your query.",
            "sections": {
                "summary": "No relevant information found.",
                "details": "",
                "analysis": "",
                "recommendations": ""
            },
            "visualization": None,
            "source_type": "aggregated",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "sources": [],
                "confidence": 0.0,
                "processing_time": 0.0,
                "result_count": 0
            }
        }
    
    def _generate_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Generate an error response structure."""
        return {
            "type": "error",
            "query": query,
            "response": f"An error occurred while processing your request: {error}",
            "sections": {
                "summary": f"Error: {error}",
                "details": "",
                "analysis": "",
                "recommendations": ""
            },
            "visualization": None,
            "source_type": "error",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "sources": [],
                "confidence": 0.0,
                "processing_time": 0.0,
                "result_count": 0,
                "error": error
            }
        }
    
    def _identify_primary_response(
        self,
        responses: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Identify the primary response based on source type."""
        if not responses:
            return None
        
        # Sort responses by source weight
        weighted_responses = []
        for response in responses:
            source_type = response.get("source_type", "unknown")
            weight = self.source_weights.get(source_type, self.source_weights[SourceType.UNKNOWN])
            weighted_responses.append((weight, response))
        
        # Return response with highest weight
        return sorted(weighted_responses, key=lambda x: x[0], reverse=True)[0][1]
    
    def _extract_section(
        self,
        responses: List[Dict[str, Any]],
        section: str
    ) -> str:
        """Extract and combine section content from responses."""
        section_content = []
        
        for response in responses:
            if content := response.get(section):
                if isinstance(content, list):
                    section_content.extend(content)
                else:
                    section_content.append(content)
        
        if not section_content:
            return ""
        
        if isinstance(section_content[0], str):
            return "\n\n".join(filter(None, section_content))
        else:
            return "\n\n".join(str(item) for item in section_content if item)
    
    def _format_final_response(
        self,
        aggregated_response: str,
        visualization_data: Optional[Dict[str, Any]] = None,
        web_search_data: Optional[List[Dict[str, Any]]] = None,
        rag_data: Optional[List[Dict[str, Any]]] = None,
        conflicts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Format the final response with all components."""
        try:
            # Add citations to the response
            response_with_citations = self._add_citations(
                aggregated_response,
                web_search_data or [],
                rag_data or []
            )
            
            # Format response sections
            sections = {
                ResponseSection.SUMMARY: self._extract_section(response_with_citations, "Summary"),
                ResponseSection.DETAILS: self._extract_section(response_with_citations, "Details"),
                ResponseSection.ANALYSIS: self._extract_section(response_with_citations, "Analysis"),
                ResponseSection.VISUALIZATION: visualization_data,
                ResponseSection.SOURCES: self._format_sources(web_search_data or [], rag_data or []),
                ResponseSection.ALERTS: self._format_alerts(conflicts or []),
                ResponseSection.RECOMMENDATIONS: self._extract_section(response_with_citations, "Recommendations"),
                ResponseSection.REFERENCES: self._generate_references(response_with_citations)
            }
            
            # Format the final response text
            response_text = []
            if sections[ResponseSection.SUMMARY]:
                response_text.append(f"## Summary\n{sections[ResponseSection.SUMMARY]}")
            if sections[ResponseSection.DETAILS]:
                response_text.append(f"## Details\n{sections[ResponseSection.DETAILS]}")
            if sections[ResponseSection.ANALYSIS]:
                response_text.append(f"## Analysis\n{sections[ResponseSection.ANALYSIS]}")
            if sections[ResponseSection.RECOMMENDATIONS]:
                response_text.append(f"## Recommendations\n{sections[ResponseSection.RECOMMENDATIONS]}")
            
            return {
                "type": "aggregated",
                "response": "\n\n".join(response_text),
                "sections": {k.value: v for k, v in sections.items() if v},
                "visualization": visualization_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error formatting final response: {str(e)}")
            return {
                "type": "error",
                "response": f"Error formatting response: {str(e)}",
                "visualization": None,
                "timestamp": datetime.now().isoformat()
            }

    def _add_citations(self, response: str,
                      web_search_data: List[Dict[str, Any]],
                      rag_data: List[Dict[str, Any]]) -> str:
        """Add citations to the response text."""
        try:
            # Create citation mapping
            citations = {}
            citation_count = 1
            
            # Add web search citations
            for item in web_search_data:
                if "title" in item and "url" in item:
                    citations[item["title"]] = {
                        "number": citation_count,
                        "url": item["url"]
                    }
                    citation_count += 1
            
            # Add RAG citations
            for item in rag_data:
                if "id" in item:
                    citations[item["id"]] = {
                        "number": citation_count,
                        "reference": f"Internal Document {item['id']}"
                    }
                    citation_count += 1
            
            # Add citations to text
            modified_response = response
            for text, citation in citations.items():
                modified_response = modified_response.replace(
                    text,
                    f"{text} [{citation['number']}]"
                )
            
            # Add references section
            if citations:
                modified_response += "\n\n## References\n"
                for text, citation in citations.items():
                    if "url" in citation:
                        modified_response += f"[{citation['number']}] {text}: {citation['url']}\n"
                    else:
                        modified_response += f"[{citation['number']}] {citation['reference']}\n"
            
            return modified_response
            
        except Exception as e:
            logger.error(f"Error adding citations: {str(e)}")
            return response

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a section from the response text."""
        try:
            pattern = f"## {section_name}\\n(.*?)(?=\\n## |$)"
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else None
        except Exception as e:
            logger.error(f"Error extracting section {section_name}: {str(e)}")
            return None

    def _format_sources(self, web_search_data: List[Dict[str, Any]],
                       rag_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Format sources section."""
        try:
            return {
                "web": web_search_data,
                "internal": rag_data
            }
        except Exception as e:
            logger.error(f"Error formatting sources: {str(e)}")
            return {}

    def _format_alerts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format alerts from conflicts."""
        try:
            alerts = []
            for conflict in conflicts:
                alerts.append({
                    "type": "conflict",
                    "topic": conflict["topic"],
                    "message": conflict["explanation"],
                    "severity": "warning"
                })
            return alerts
        except Exception as e:
            logger.error(f"Error formatting alerts: {str(e)}")
            return []

    def _generate_references(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate references from response sources."""
        references = []
        ref_counter = 1
        
        for response in responses:
            if source := response.get("source"):
                references.append({
                    "id": ref_counter,
                    "source": source,
                    "type": response.get("type", "unknown"),
                    "timestamp": response.get("timestamp", datetime.now().isoformat())
                })
                ref_counter += 1
        
        return references

    def _extract_visualization_data(self, responses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract visualization data from responses.
        
        Args:
            responses: List of responses from different agents
            
        Returns:
            Optional[Dict[str, Any]]: Visualization data if available
        """
        for response in responses:
            if response.get("type") == "analytics" and "visualization" in response:
                return {
                    "visualization": response.get("visualization"),
                    "analytics_params": response.get("analytics_params", {})
                }
        return None
    
    def _extract_web_search_data(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract web search data from responses.
        
        Args:
            responses: List of responses from different agents
            
        Returns:
            List[Dict[str, Any]]: Web search data if available
        """
        web_search_data = []
        for response in responses:
            if response.get("type") == "web_news_search" and "sources" in response:
                web_search_data.append({
                    "sources": response.get("sources", []),
                    "search_type": response.get("search_type", "")
                })
        return web_search_data
    
    def _extract_rag_data(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract RAG data from responses.
        
        Args:
            responses: List of responses from different agents
            
        Returns:
            List[Dict[str, Any]]: RAG data if available
        """
        rag_data = []
        for response in responses:
            if response.get("type") == "rag" and "retrieved_data" in response:
                rag_data.append({
                    "retrieved_data": response.get("retrieved_data", []),
                    "namespaces": response.get("namespaces", [])
                })
        return rag_data

    def aggregate_responses(self, query: str, primary_response: Dict[str, Any], 
                          secondary_responses: List[Dict[str, Any]], routing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate responses with improved handling for UI pipeline."""
        try:
            start_time = time.time()
            
            # Validate and normalize primary response
            if not primary_response:
                logger.error("Primary response is None")
                return self._generate_empty_response(query)
            
            if isinstance(primary_response, str):
                try:
                    primary_response = json.loads(primary_response)
                except json.JSONDecodeError:
                    primary_response = {
                        "summary": primary_response,
                        "source_type": "unknown"
                    }
            
            if not isinstance(primary_response, dict):
                logger.error(f"Invalid primary response type: {type(primary_response)}")
                return self._generate_empty_response(query)
            
            # Normalize secondary responses
            normalized_secondary = []
            for response in secondary_responses:
                if isinstance(response, str):
                    try:
                        response = json.loads(response)
                    except json.JSONDecodeError:
                        response = {
                            "summary": response,
                            "source_type": "unknown"
                        }
                if isinstance(response, dict):
                    normalized_secondary.append(response)
                else:
                    logger.warning(f"Skipping invalid secondary response: {type(response)}")
            
            # Extract main content from primary response
            main_content = primary_response.get("summary", "") or primary_response.get("details", "") or primary_response.get("response", "")
            if not main_content:
                logger.warning("No main content in primary response")
                if normalized_secondary:
                    # Try to use content from secondary responses
                    for response in normalized_secondary:
                        if content := (response.get("summary") or response.get("details") or response.get("response")):
                            main_content = content
                            break
                if not main_content:
                    return self._generate_empty_response(query)
            
            # Process visualization data if available
            visualization_data = None
            for response in [primary_response] + normalized_secondary:
                if viz := response.get("visualization"):
                    visualization_data = viz
                    break
            
            # Handle structured data based on query category
            category = routing_info.get("category", "").lower() if routing_info else ""
            structured_data = None
            
            if "inventory" in category:
                inventory_responses = [r for r in [primary_response] + normalized_secondary 
                                    if r.get("type") == "inventory" or r.get("source_type") == "rag"]
                if inventory_responses:
                    structured_data = self._format_structured_data(inventory_responses, "inventory")
            elif "transport" in category:
                transport_responses = [r for r in [primary_response] + normalized_secondary 
                                    if r.get("type") == "transport" or r.get("source_type") == "rag"]
                if transport_responses:
                    structured_data = self._format_structured_data(transport_responses, "transport")
            
            # Extract sections from primary response
            sections = {
                "summary": primary_response.get("summary", "") or main_content,
                "details": structured_data if structured_data else primary_response.get("details", ""),
                "analysis": primary_response.get("analysis", ""),
                "recommendations": primary_response.get("recommendations", "")
            }
            
            # Add web search context if available
            web_responses = [r for r in normalized_secondary if r.get("source_type") == "web_news_search"]
            if web_responses:
                web_context = self._format_web_context(web_responses)
                if web_context:
                    sections["analysis"] = sections["analysis"] + "\n\n### Market Context\n" + web_context if sections["analysis"] else web_context
            
            # Combine all responses
            combined_response = {
                "type": "aggregated",
                "query": query,
                "summary": sections["summary"],
                "details": sections["details"],
                "analysis": sections["analysis"],
                "recommendations": sections["recommendations"],
                "visualization": visualization_data,
                "source_type": "aggregated",
                "confidence": primary_response.get("confidence", 0.0),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "routing_category": category,
                    "primary_source": primary_response.get("source_type", "unknown"),
                    "secondary_sources": [r.get("source_type", "unknown") for r in normalized_secondary],
                    "result_count": len([primary_response] + normalized_secondary)
                }
            }
            
            # Add UI-specific formatting
            combined_response["ui_format"] = {
                "pipeline_status": "completed",
                "has_visualization": bool(visualization_data),
                "has_structured_data": bool(structured_data),
                "sections": [
                    {"name": "Summary", "content": sections["summary"]},
                    {"name": "Details", "content": sections["details"]},
                    {"name": "Analysis", "content": sections["analysis"]},
                    {"name": "Recommendations", "content": sections["recommendations"]}
                ]
            }
            
            return combined_response
            
        except Exception as e:
            logger.error(f"Error aggregating responses: {str(e)}")
            return self._generate_error_response(query, str(e))
    
    def _format_structured_data(self, data: List[Dict[str, Any]], category: str) -> str:
        """Format structured data into a readable response."""
        if not data:
            return "No data available."
        
        if category == "inventory":
            # Format inventory data
            items = []
            for item in data:
                item_str = f"- {item.get('GenericName', 'Unknown Item')}:\n"
                item_str += f"  Current Stock: {item.get('CurrentStock', 'N/A')}\n"
                item_str += f"  Reorder Point: {item.get('ReorderPoint', 'N/A')}\n"
                if item.get('ExpiryDate'):
                    item_str += f"  Expiry Date: {item['ExpiryDate']}\n"
                items.append(item_str)
            
            return "Here's the inventory information:\n\n" + "\n".join(items)
            
        elif category == "transport":
            # Format transport data
            shipments = []
            for shipment in data:
                ship_str = f"- Shipment {shipment.get('ShipmentID', 'Unknown')}:\n"
                ship_str += f"  Status: {shipment.get('Status', 'N/A')}\n"
                ship_str += f"  Origin: {shipment.get('Origin', 'N/A')}\n"
                ship_str += f"  Destination: {shipment.get('Destination', 'N/A')}\n"
                if shipment.get('ETA'):
                    ship_str += f"  ETA: {shipment['ETA']}\n"
                shipments.append(ship_str)
            
            return "Here's the transport information:\n\n" + "\n".join(shipments)
        
        return "Data available but format not recognized."

    def _format_web_context(self, web_data: List[Dict[str, Any]]) -> str:
        """Format web search data into market context."""
        if not web_data:
            return ""
        
        context_parts = []
        for data in web_data:
            if content := data.get("content"):
                context_parts.append(content)
            elif summary := data.get("summary"):
                context_parts.append(summary)
        
        if not context_parts:
            return ""
        
        return "\n\n".join(context_parts)

    def _process_visualizations(self, viz_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and enhance visualizations"""
        processed_viz = {}
        
        for viz in viz_data:
            for viz_type, viz_content in viz.items():
                try:
                    # Parse the JSON string back to a figure
                    if isinstance(viz_content, str):
                        fig_dict = json.loads(viz_content)
                        fig = go.Figure(fig_dict)
                        
                        # Enhance the figure
                        fig.update_layout(
                            template="plotly_white",
                            title_font_size=16,
                            showlegend=True,
                            margin=dict(t=50, l=50, r=50, b=50)
                        )
                        
                        processed_viz[viz_type] = fig.to_json()
                except Exception as e:
                    continue
        
        return processed_viz

    def _format_data_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format and structure data insights"""
        insights = {}
        
        if "summary_stats" in data:
            insights["statistical_summary"] = data["summary_stats"]
        
        if "correlations" in data:
            # Filter significant correlations
            corr_data = pd.DataFrame(data["correlations"])
            significant_corr = corr_data[abs(corr_data) > 0.5].unstack()
            significant_corr = significant_corr[significant_corr != 1.0]
            insights["significant_correlations"] = significant_corr.to_dict()
        
        if "trends" in data:
            insights["key_trends"] = [
                trend for trend in data["trends"]
                if trend["trend"]["confidence"] > 0.8
            ]
        
        return insights

    async def process_outputs(
        self,
        agent_responses: Dict[str, Dict[str, Any]],
        query: str,
        routing_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process outputs from multiple agents into a unified response.
        
        Args:
            agent_responses: Dictionary of agent responses
            query: The original user query
            routing_info: Information about how the query was routed
            
        Returns:
            A dictionary containing the unified response
        """
        try:
            logger.info(f"Processing outputs for query: {query}")
            
            if not agent_responses:
                logger.warning("No agent responses to process")
                return {
                    "response": "I'm sorry, but I couldn't find any relevant information to answer your question.",
                    "source_type": "aggregator",
                    "query": query
                }
            
            # Log the agent responses for debugging
            for agent_type, response in agent_responses.items():
                if isinstance(response, dict):
                    logger.debug(f"Response from {agent_type}: {list(response.keys())}")
                    # Ensure all responses have a "response" key for consistency
                    if "response" not in response and "content" in response:
                        response["response"] = response["content"]
                else:
                    logger.debug(f"Response from {agent_type} is not a dictionary: {type(response)}")
                    # Convert string responses to dictionary
                    if isinstance(response, str):
                        agent_responses[agent_type] = {"response": response, "source_type": agent_type}
            
            # Convert agent responses for the prompt
            agent_responses_text = "\n\n".join([
                f"{agent_type.upper()}:\n{response.get('response', str(response))}"
                for agent_type, response in agent_responses.items()
            ])
            
            # Use the language model to create a unified response
            prompt = self._create_aggregation_prompt()
            prompt_text = prompt.format(query=query, agent_responses=agent_responses_text)
            
            messages = [
                {"role": "system", "content": "You are a expert-level aggregator for pharmaceutical supply chain management. You create comprehensive, well-structured responses by combining information from multiple specialized agents."},
                {"role": "user", "content": prompt_text}
            ]
            
            response_text = await self.get_completion(messages)
            
            # Parse the structured response
            parsed_response = self._parse_llm_response(response_text)
            parsed_response["query"] = query
            parsed_response["source_type"] = "aggregator"
            
            # Add source information
            source_info = self._extract_source_info(agent_responses)
            if source_info:
                parsed_response["sources"] = source_info
            
            # Add metadata from routing
            if routing_info:
                parsed_response["metadata"] = routing_info
            
            # Add agent_responses for reference
            parsed_response["agent_responses"] = agent_responses
            
            # Ensure a response key exists
            if "response" not in parsed_response:
                if "summary" in parsed_response:
                    parsed_response["response"] = parsed_response["summary"]
                elif "details" in parsed_response:
                    parsed_response["response"] = parsed_response["details"]
                else:
                    parsed_response["response"] = response_text
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error in process_outputs: {str(e)}")
            return {
                "response": f"I encountered an error processing the responses: {str(e)}",
                "query": query,
                "source_type": "aggregator",
                "error": True
            }

    def format_for_ui(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format the response specifically for UI consumption"""
        try:
            logger.info(f"Formatting for UI. Response keys: {response.keys()}")
            
            # Basic UI response structure
            ui_response = {
                "content": response.get("response", "No response content available"),
                "type": "assistant",
                "timestamp": datetime.now().isoformat(),
                "pipeline_status": {
                    "query_received": "completed",
                    "query_enhancement": "completed",
                    "router_agent": "completed",
                    "knowledge_base": "completed",
                    "response_generation": "completed",
                    "ui_response": "completed"
                }
            }
            
            # Add summary if available
            if "summary" in response and response["summary"]:
                ui_response["summary"] = response["summary"]
            
            # Add details if available
            if "details" in response and response["details"]:
                ui_response["details"] = response["details"]
            
            # Add analysis if available
            if "analysis" in response and response["analysis"]:
                ui_response["analysis"] = response["analysis"]
            
            # Add recommendations if available
            if "recommendations" in response and response["recommendations"]:
                ui_response["recommendations"] = response["recommendations"]
            
            # Add sources if available
            if "sources" in response and response["sources"]:
                ui_response["sources"] = response["sources"]
            
            # Add visualization if available
            if "visualization" in response and response["visualization"]:
                ui_response["visualization"] = response["visualization"]
            
            # Add table data if available
            if "table_data" in response and response["table_data"]:
                ui_response["table_data"] = response["table_data"]
            
            # Add session ID if available
            if "query" in response:
                ui_response["session_id"] = str(uuid.uuid4())
            
            logger.info(f"Formatted UI response keys: {ui_response.keys()}")
            return ui_response
            
        except Exception as e:
            logger.error(f"Error formatting response for UI: {str(e)}")
            return {
                "content": "Sorry, there was an error formatting the response for display.",
                "type": "error",
                "timestamp": datetime.now().isoformat()
            }

    def _format_insights_for_ui(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format insights into a UI-friendly structure"""
        formatted_insights = []
        
        for category, data in insights.items():
            if isinstance(data, dict):
                formatted_insights.append({
                    "category": category.replace("_", " ").title(),
                    "type": "table",
                    "data": self._dict_to_table(data)
                })
            elif isinstance(data, list):
                formatted_insights.append({
                    "category": category.replace("_", " ").title(),
                    "type": "list",
                    "data": data
                })
        
        return formatted_insights

    def _format_visualizations_for_ui(self, visualizations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format visualizations for UI rendering"""
        formatted_viz = []
        
        for viz_type, viz_data in visualizations.items():
            formatted_viz.append({
                "type": viz_type,
                "title": viz_type.replace("_", " ").title(),
                "data": viz_data,
                "config": {
                    "responsive": True,
                    "displayModeBar": True,
                    "displaylogo": False
                }
            })
        
        return formatted_viz

    def _dict_to_table(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary data to table format"""
        if not data:
            return {"headers": [], "rows": []}
            
        df = pd.DataFrame(data)
        return {
            "headers": df.columns.tolist(),
            "rows": df.values.tolist()
        }

    def _extract_title(self, url: str) -> str:
        """Extract a readable title from URL"""
        try:
            return url.split("/")[-1].replace("-", " ").replace("_", " ").title()
        except:
            return url

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured sections.
        
        Args:
            response_text: The raw text response from the language model
            
        Returns:
            A dictionary with structured sections
        """
        # Initialize sections
        sections = {
            "summary": "",
            "details": "",
            "analysis": "",
            "recommendations": ""
        }
        
        # Extract sections using markdown-style headers
        current_section = "summary"  # Default section if no headers are found
        current_content = []
        
        for line in response_text.split('\n'):
            if line.startswith('## Summary') or line.startswith('# Summary'):
                current_section = "summary"
                current_content = []
            elif line.startswith('## Details') or line.startswith('# Details'):
                # Save previous section content
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = "details"
                current_content = []
            elif line.startswith('## Analysis') or line.startswith('# Analysis'):
                # Save previous section content
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = "analysis"
                current_content = []
            elif line.startswith('## Recommendations') or line.startswith('# Recommendations'):
                # Save previous section content
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = "recommendations"
                current_content = []
            elif not line.startswith('##') and not line.startswith('#'):
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # If no sections were found, put everything in summary
        if not any(sections.values()):
            sections["summary"] = response_text.strip()
        
        return sections
    
    def _extract_source_info(self, agent_responses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source information from agent responses.
        
        Args:
            agent_responses: Dictionary of agent responses
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        
        for agent_type, response in agent_responses.items():
            if not isinstance(response, dict):
                continue
                
            source_type = response.get("source_type", agent_type)
            confidence = 0.8  # Default confidence
            
            if source_type == "rag":
                sources.append({
                    "type": "rag",
                    "source": "Internal Database",
                    "confidence": 0.95
                })
            elif source_type == "web_search":
                sources.append({
                    "type": "web_search",
                    "source": "Web Search Results",
                    "confidence": 0.7
                })
            elif source_type == "cag":
                sources.append({
                    "type": "cag",
                    "source": "Internal Documents",
                    "confidence": 0.9
                })
            elif source_type == "enhanced_analytics":
                sources.append({
                    "type": "analytics",
                    "source": "Data Analytics",
                    "confidence": 0.85
                })
        
        return sources

    async def get_completion(self, messages: List[Dict[str, str]]) -> str:
        """Get a completion from the language model asynchronously."""
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}")
            return f"Error: {str(e)}"

# For direct testing
if __name__ == "__main__":
    agent = AggregatorAgent()
    test_responses = [
        {
            "type": "rag",
            "response": "Current stock level of Aspirin is 8,500 units",
            "confidence": 0.95,
            "data": {"ItemID": "ASP001", "CurrentStock": 8500}
        },
        {
            "type": "web_news_search",
            "response": "Recent reports indicate Aspirin production increased by 5%",
            "sources": [{"title": "Pharma Report", "url": "https://example.com/report"}]
        },
        {
            "type": "analytics",
            "response": "Stock levels show upward trend",
            "visualization": "base64_encoded_chart"
        }
    ]
    
    result = agent.process_responses("What is the current stock level of Aspirin?", test_responses)
    print("\nAggregated Response:")
    print(f"Type: {result['type']}")
    print(f"Response:\n{result['response']}")
    if "sections" in result:
        print("\nSections:")
        for section, content in result["sections"].items():
            print(f"\n{section.upper()}:")
            print(content) 