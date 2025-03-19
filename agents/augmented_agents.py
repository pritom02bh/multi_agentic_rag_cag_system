from typing import Dict, Any, Optional, List, Tuple
import logging
import json
import os
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
import requests
from .base_agent import BaseAgent, AgentType, AgentConfig
from pinecone import Pinecone as PineconeClient
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMChain
import openai
from openai import OpenAI
import aiohttp

logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 384
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

class BaseVectorAgent(BaseAgent):
    """Base class for agents that use vector search."""
    
    def __init__(self, config: AgentConfig, pinecone_client: Any, index_name: str, namespaces: Optional[List[str]] = None):
        super().__init__(config)
        self.namespaces = namespaces or []
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            dimensions=EMBEDDING_DIMENSION
        )
        try:
            # Get the index from the client to inspect document structure
            index = pinecone_client.Index(index_name)
            query_embedding = self.embeddings.embed_query("test")
            raw_results = index.query(
                vector=query_embedding,
                top_k=1,
                namespace=self.namespaces[0] if self.namespaces else None,
                include_metadata=True
            )
            if raw_results['matches']:
                metadata = raw_results['matches'][0]['metadata']
                # Find the most suitable text key
                text_keys = ['CargoDescription', 'GenericName', 'OriginLocationName', 'DestinationLocationName']
                text_key = next((k for k in text_keys if k in metadata), text_keys[0])
                logger.info(f"Using {text_key} as the primary text key")
                
                self.vectorstore = PineconeVectorStore.from_existing_index(
                    index_name=index_name,
                    embedding=self.embeddings,
                    text_key=text_key,
                    namespace=self.namespaces[0] if self.namespaces else None
                )
                logger.info(f"Successfully initialized vector store with index {index_name}")
            else:
                raise ValueError("No documents found in the index")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def _create_document_text(self, metadata: dict) -> str:
        """Create a text representation from document metadata."""
        text_fields = []
        
        # Add cargo information if available
        cargo_info = []
        if metadata.get('CargoDescription'):
            cargo_info.append(metadata['CargoDescription'])
        if metadata.get('GenericName'):
            cargo_info.append(f"Generic Name: {metadata['GenericName']}")
        if cargo_info:
            text_fields.append(f"Cargo: {' - '.join(cargo_info)}")
        
        # Add location information
        origin = f"{metadata.get('OriginLocationName', '')}, {metadata.get('OriginCountry', '')}"
        if origin.strip(', '):
            text_fields.append(f"Origin: {origin}")
        
        destination = f"{metadata.get('DestinationLocationName', '')}, {metadata.get('DestinationCountry', '')}"
        if destination.strip(', '):
            text_fields.append(f"Destination: {destination}")
        
        # Add transport information
        transport_info = []
        if metadata.get('ModeOfTransport'):
            transport_info.append(metadata['ModeOfTransport'])
        if metadata.get('Carrier'):
            transport_info.append(metadata['Carrier'])
        if transport_info:
            text_fields.append(f"Transport: {' - '.join(transport_info)}")
        
        # Add temperature information if available
        temp_info = []
        if metadata.get('TemperatureRange'):
            temp_info.append(metadata['TemperatureRange'])
        if metadata.get('TemperatureCategory'):
            temp_info.append(metadata['TemperatureCategory'])
        if temp_info:
            text_fields.append(f"Temperature: {' - '.join(temp_info)}")
        
        # Add quality information if available
        if metadata.get('QualityMetrics'):
            text_fields.append(f"Quality Metrics: {metadata['QualityMetrics']}")
        if metadata.get('QualityRequirements'):
            text_fields.append(f"Quality Requirements: {metadata['QualityRequirements']}")
        
        # Add compliance information if available
        if metadata.get('ComplianceRequirements'):
            text_fields.append(f"Compliance: {metadata['ComplianceRequirements']}")
        if metadata.get('RiskCategory'):
            text_fields.append(f"Risk Category: {metadata['RiskCategory']}")
        
        return "\n".join(text_fields) if text_fields else str(metadata)

    def _search_namespace(self, query: str, namespace: str, k: int = 4) -> List[Document]:
        """Search within a specific namespace."""
        try:
            # Update namespace for the existing vectorstore
            self.vectorstore.namespace = namespace
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                namespace=namespace
            )
            # Transform results to include full text representation
            transformed_results = []
            for doc, score in results:
                doc.page_content = self._create_document_text(doc.metadata)
                transformed_results.append((doc, score))
            return transformed_results
        except Exception as e:
            logger.error(f"Error searching namespace {namespace}: {e}")
            return []

    async def search_documents(self, query: str, namespaces: List[str], k: int = 3) -> List[Tuple[Document, float]]:
        """Search for relevant documents across specified namespaces."""
        all_results = []
        for namespace in namespaces:
            try:
                results = self._search_namespace(query, namespace, k)
                if results:
                    all_results.extend(results)
            except Exception as e:
                logger.error(f"Error querying namespace {namespace}: {e}")
                continue
        
        # Sort by score in descending order
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]
    
    async def get_completion(self, messages: List[Dict[str, str]]) -> str:
        """Get completion from OpenAI."""
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature or DEFAULT_TEMPERATURE,
                max_tokens=self.config.max_tokens or DEFAULT_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}")
            raise

class RAGAgent(BaseVectorAgent):
    """Retrieval Augmented Generation agent for medical supply chain queries."""
    
    def __init__(self, config: AgentConfig, pinecone_client: Any, index_name: str, namespaces: Optional[List[str]] = None):
        super().__init__(config, pinecone_client, index_name, namespaces or ['transport', 'guidelines', 'inventory', 'policies'])
        self.last_context = None
    
    def get_type(self) -> AgentType:
        return AgentType.RAG
    
    async def process(self, query: str) -> str:
        try:
            # Search for relevant documents
            results = await self.search_documents(query, self.namespaces)
            
            if not results:
                self.last_context = None
                return {
                    "response": "I apologize, but I couldn't find any relevant information in the knowledge base.",
                    "content": "I apologize, but I couldn't find any relevant information in the knowledge base.",
                    "source_type": "rag",
                    "error": False
                }
            
            # Prepare context from results
            # The results are tuples (document, score) where document contains text and metadata
            context_items = []
            for result in results:
                if isinstance(result, tuple) and len(result) >= 2:
                    doc_text = result[0]
                    namespace = "unknown"
                    source = "unknown"
                    
                    # Check if metadata is available in different formats
                    if hasattr(doc_text, 'metadata'):
                        metadata = doc_text.metadata
                        namespace = metadata.get('namespace', 'unknown')
                        source = metadata.get('source', 'unknown')
                    elif isinstance(doc_text, dict) and 'metadata' in doc_text:
                        metadata = doc_text['metadata']
                        namespace = metadata.get('namespace', 'unknown')
                        source = metadata.get('source', 'unknown')
                        
                    # Get the actual text content
                    content = ""
                    if hasattr(doc_text, 'page_content'):
                        content = doc_text.page_content
                    elif isinstance(doc_text, dict) and 'page_content' in doc_text:
                        content = doc_text['page_content']
                    elif isinstance(doc_text, str):
                        content = doc_text
                    
                    context_items.append(f"From {source} ({namespace}):\n{content}")
            
            context = "\n\n".join(context_items)
            
            # Store the context for potential use in UI formatting
            self.last_context = context
            
            # Generate response using the context
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": f"""Based on the following information from our medical supply chain database:

{context}

Please answer this question: {query}

Provide a clear and concise response that directly addresses the query while incorporating relevant information from the sources."""}
            ]
            
            # Get completion from model
            completion = await self.get_completion(messages)
            
            # Construct response with context included
            response = {
                "response": completion,
                "content": completion,
                "source_type": "rag",
                "context": context,
                "sources": [{"type": "database", "name": "inventory"}],
                "error": False
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}")
            self.last_context = None
            return {
                "response": f"I encountered an error retrieving information: {str(e)}",
                "content": f"I encountered an error retrieving information: {str(e)}",
                "source_type": "rag",
                "error": True
            }

class WebSearchAgent(BaseAgent):
    """Web Search Agent for real-time information."""
    
    def __init__(self, config: AgentConfig, search_api_key: str):
        super().__init__(config)
        self.search_api_key = search_api_key
    
    def get_type(self) -> AgentType:
        return AgentType.WEB_SEARCH
    
    async def process(self, query: str) -> str:
        """Process a query using web search."""
        try:
            articles = await self._search_articles(query)
            
            if not articles:
                return "No relevant information found."
            
            # Prepare context from articles
            context = "\n".join(f"Title: {article['title']}\nDescription: {article['description']}"
                              for article in articles)
            
            # Generate response using OpenAI
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": f"""Based on these recent articles:

{context}

Please provide a comprehensive answer to: {query}

Include relevant information from the articles and cite your sources."""}
            ]
            
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature or DEFAULT_TEMPERATURE,
                max_tokens=self.config.max_tokens or DEFAULT_MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in Web Search agent: {str(e)}")
            return f"I encountered an error while searching: {str(e)}"
    
    async def _search_articles(self, query: str, page_size: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant articles using NewsAPI."""
        try:
            search_url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": self.search_api_key,
                "sortBy": "relevancy",
                "pageSize": page_size
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"NewsAPI error: {response.status}")
                        return []
                    
                    data = await response.json()
                    return data.get("articles", [])
        except Exception as e:
            logger.error(f"Error searching articles: {str(e)}")
            return []

class EnhancedAnalyticsAgent(BaseVectorAgent):
    """Enhanced Analytics Agent for data visualization and analysis."""
    
    def __init__(self, config: AgentConfig, pinecone_client: Any, index_name: str, namespaces: Optional[List[str]] = None):
        super().__init__(config, pinecone_client, index_name, namespaces or ['transport', 'inventory'])
    
    def get_type(self) -> AgentType:
        return AgentType.ENHANCED_ANALYTICS
    
    async def process(self, query: str) -> str:
        try:
            # Search for relevant documents
            results = await self.search_documents(query, self.namespaces)
            
            if not results:
                return "I apologize, but I couldn't find any relevant data for analysis."
            
            # Extract data points and metadata
            data_points = []
            for doc in results:
                try:
                    if 'data' in doc.metadata:
                        data_point = {
                            'data': doc.metadata['data'],
                            'source': doc.metadata.get('source', 'unknown'),
                            'namespace': doc.metadata.get('namespace', 'unknown'),
                            'timestamp': doc.metadata.get('timestamp', 'unknown')
                        }
                        data_points.append(data_point)
                except Exception as e:
                    logger.warning(f"Error processing data point: {str(e)}")
            
            if not data_points:
                return "I found relevant documents, but they don't contain structured data for analysis."
            
            # Generate analysis using the data points
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": f"""Based on the following data points from our medical supply chain:

{json.dumps(data_points, indent=2)}

Please provide a detailed analysis addressing this query: {query}

Include:
1. Key trends or patterns
2. Notable statistics
3. Relevant comparisons
4. Actionable insights
5. Suggested visualization type: {self._determine_visualization_type(query)}"""}
            ]
            
            return await self.get_completion(messages)
            
        except Exception as e:
            logger.error(f"Error in Enhanced Analytics processing: {str(e)}")
            return f"I encountered an error while analyzing the data: {str(e)}"
    
    def _determine_visualization_type(self, query: str) -> str:
        """Determine the most appropriate visualization type based on the query."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["trend", "over time", "historical"]):
            return "line_chart"
        elif any(word in query_lower for word in ["compare", "comparison", "versus"]):
            return "bar_chart"
        elif any(word in query_lower for word in ["distribution", "spread"]):
            return "histogram"
        elif any(word in query_lower for word in ["relationship", "correlation"]):
            return "scatter_plot"
        else:
            return "table"

class CAGAgent(BaseAgent):
    """Collaborative Agent Group that coordinates multiple agents."""
    
    def __init__(self, config: Optional[AgentConfig] = None, agents: Optional[Dict[str, BaseAgent]] = None):
        super().__init__(config)
        self.agents = agents or {}
        self.openai_client = None
        try:
            # Initialize OpenAI client for combined responses
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
    
    def get_type(self) -> AgentType:
        return AgentType.CAG
    
    async def process(self, query: str) -> str:
        """Process the query by coordinating multiple agents."""
        try:
            # Parse the query if it's a JSON string
            if query.startswith('{') and query.endswith('}'):
                try:
                    responses = json.loads(query)
                    return self._combine_responses(responses)
                except json.JSONDecodeError:
                    pass  # Not a JSON string, process as normal query
            
            # Process as a normal query
            responses = await self._route_query(query)
            return self._combine_responses(responses)
            
        except Exception as e:
            logger.error(f"Error in CAG processing: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def _route_query(self, query: str) -> Dict[str, str]:
        """Route the query to appropriate agents based on content."""
        responses = {}
        query_lower = query.lower()
        
        # Use RAG agent for knowledge base queries
        if "stock" in query_lower or "inventory" in query_lower:
            if "rag" in self.agents:
                responses["rag"] = await self.agents["rag"].process(query)
        
        # Use web search for current events/news
        if "news" in query_lower or "current" in query_lower:
            if "web_search" in self.agents:
                responses["web_search"] = await self.agents["web_search"].process(query)
        
        # Use enhanced analytics for analysis requests
        if "analyze" in query_lower or "report" in query_lower:
            if "enhanced_analytics" in self.agents:
                responses["enhanced_analytics"] = await self.agents["enhanced_analytics"].process(query)
        
        # If no specific agents were selected, use RAG as default
        if not responses and "rag" in self.agents:
            responses["rag"] = await self.agents["rag"].process(query)
        
        return responses
    
    def _combine_responses(self, responses: Dict[str, str]) -> str:
        """Combine responses from multiple agents."""
        combined_response = ""
        for agent_name, response in responses.items():
            if response:
                combined_response += f"\n{agent_name.upper()} Response:\n{response}\n"
        
        return combined_response.strip() if combined_response else "I apologize, but I couldn't find relevant information to answer your query." 