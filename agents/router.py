import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Set
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import AppConfig
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import re
import traceback
from .transport_knowledge import TransportKnowledgeBase
from .web_search import WebSearchAgent
from utils.chroma_manager import ChromaDBManager
from utils.chroma_parser import ChromaParser

logger = logging.getLogger(__name__)

class DataRouter:
    """Routes queries to appropriate data sources and loads relevant data."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize the router with configuration."""
        self.config = config or AppConfig()
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            logger.info("DataRouter initialized with OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
            
        # Define data sources paths
        self.data_sources = {
            'inventory': 'embedded_data/inventory_data_embedded.csv',
            'transport': 'embedded_data/transport_history_embedded.csv',
            'guidelines': 'embedded_data/us_government_guidelines_for_medicine_transportation_and_storage_embedded.csv',
            'policy': 'embedded_data/inventory_management_policy_embedded.csv',
            'web': 'web_search'  # Add web search as a source
        }
        
        # Initialize ChromaDB manager
        self.chroma_manager = ChromaDBManager(config)
        
        # Initialize ChromaParser
        self.chroma_parser = getattr(config, 'chroma_parser', None)
        if not self.chroma_parser:
            self.chroma_parser = ChromaParser(base_dir="vector_db_separate")
            logger.info("Initialized ChromaParser")
        
        # Initialize transport knowledge base
        self.transport_kb = TransportKnowledgeBase()
        
        # Initialize web search agent
        self.web_search_agent = WebSearchAgent(config)
        
        # Define source-specific required columns
        self.source_columns = {
            'inventory': ['ItemID', 'GenericName', 'CurrentStock', 'ReorderPoint', 'LeadTimeDays'],
            'transport': ['ShipmentID', 'ItemID', 'Status', 'DepartureDate', 'EstimatedArrival', 'ActualArrival', 'Delay'],
            'guidelines': ['RegulationID', 'Category', 'Requirement', 'Impact'],
            'policy': ['PolicyID', 'Category', 'Threshold', 'Action'],
            'web': []  # Web search doesn't have fixed columns
        }
        
        # Keywords for source matching
        self.source_keywords = {
            'inventory': [
                'stock', 'inventory', 'level', 'reorder', 'supply', 'available', 'quantity', 
                'expiry', 'expired', 'batch', 'lot', 'shelf life', 'units', 'shortage',
                'item', 'product', 'medicine', 'drug', 'pharmaceutical', 'turnover', 'ratio'
            ],
            'transport': [
                'shipment', 'delivery', 'delay', 'transit', 'route', 'transport', 'shipping',
                'carrier', 'logistics', 'freight', 'tracking', 'temperature', 'cold chain',
                'arrival', 'departure', 'origin', 'destination', 'vehicle', 'distribution',
                'excursion', 'missing', 'temperature log', 'documentation', 'port', 'carbon footprint'
            ],
            'guidelines': [
                'regulation', 'guideline', 'requirement', 'compliance', 'standard', 'protocol',
                'legal', 'fda', 'inspection', 'safety', 'quality', 'storage condition', 
                'government', 'authority', 'recommendation', 'procedure', 'handling',
                'phmsa', 'tsa', 'un 3373', 'triple packaging', 'labeling', 'temperature requirement'
            ],
            'policy': [
                'policy', 'procedure', 'threshold', 'rule', 'protocol', 'process',
                'company policy', 'internal', 'management', 'workflow', 'approval',
                'authorization', 'responsibility', 'sop', 'operations', 'safety stock',
                'service level', 'cycle count', 'replenishment', 'classification', 'ai forecasting'
            ],
            'web': [
                'web', 'internet', 'search', 'online', 'external', 'news', 'latest', 'update',
                'current', 'market', 'competitor', 'industry', 'trend', 'research', 'article',
                'publication', 'study', 'report', 'finding', 'outside', 'global', 'worldwide',
                'updates', 'tariff', 'tariffs', 'costs', 'pricing', 'price changes',
                'headlines', 'breaking', 'developments', 'recent changes', 'current news',
                'new details', 'new information', 'new developments', 'new changes',
                'current updates', 'current events', 'recent updates', 'recent news',
                'today', 'this week', 'this month', 'this year', 'ongoing', 'emerging',
                'press release', 'announcement', 'bulletin', 'notification', 'alert'
            ]
        }
        
        # Define medication-to-source mapping for specific medications
        self.medication_sources = {
            'paracetamol': ['inventory', 'guidelines'],
            'aspirin': ['inventory', 'guidelines'],
            'ibuprofen': ['inventory', 'guidelines'], 
            'insulin': ['inventory', 'guidelines', 'transport'],
            'metformin': ['inventory', 'guidelines'],
            'lisinopril': ['inventory', 'policy'],
            'atorvastatin': ['inventory', 'transport'],
            'amoxicillin': ['inventory', 'guidelines'],
            'ciprofloxacin': ['inventory', 'guidelines'],
            'albuterol': ['inventory', 'policy', 'guidelines'],
            'adalimumab': ['inventory', 'transport'],
            'rituximab': ['inventory', 'guidelines'],
            'erythropoietin': ['inventory', 'policy'],
            'sildenafil': ['inventory', 'guidelines'],
            'flu vaccine': ['inventory', 'guidelines', 'transport'],
            'alpha interferon': ['inventory', 'guidelines'],
            'beta interferon': ['inventory', 'guidelines', 'policy'],
            'growth hormone': ['inventory', 'guidelines', 'policy'],
            'hydrocortisone': ['inventory', 'transport']
        }
        
        # Define query pattern to agent mapping
        self.query_to_agent_mapping = {
            # Patterns for Vector RAG Agent
            'vector_rag': [
                r'(what|which|list)\s+(is|are)\s+the\s+(packaging|temperature|regulations|labeling|requirements)',
                r'does\s+.+\s+require',
                r'is\s+.+\s+required',
                r'what\s+(is|are)\s+the\s+(safety\s+stock|service\s+level|reorder\s+point|lead\s+time)\s+',
                r'what\s+performance\s+metrics',
                r'what\s+role\s+does',
                r'what\s+penalties',
                r'is\s+.+\s+subject\s+to',
                r'which\s+items\s+fall\s+under',
                r'what\s+packaging\s+is\s+required',
                r'are\s+special\s+handling',
                r'what\s+compliance\s+metrics',
                r'what\s+tsa\s+regulations',
                r'what\s+are\s+the\s+quarantine',
            ],
            # Patterns for Analytics Agent
            'analytics': [
                r'how\s+many',
                r'compare',
                r'trend',
                r'which\s+port',
                r'what\s+is\s+the\s+current\s+stock\s+level',
                r'what\s+is\s+the\s+handling\s+cost\s+breakdown',
                r'what\s+is\s+the\s+(insurance\s+premium|average\s+carbon\s+footprint)',
                r'how\s+much\s+total',
                r'which\s+.+\s+are\s+running\s+low',
                r'how\s+does\s+the\s+unit\s+cost\s+.+\s+compare',
                r'which\s+shipments\s+.+\s+had\s+(missing|documentation)',
                r'what\'s\s+the\s+current\s+fill\s+rate',
                r'what\s+is\s+the\s+inventory\s+turnover\s+ratio',
                r'list\s+all\s+critical\s+stock',
            ],
            # Patterns for Hybrid (multiple agents)
            'hybrid': [
                r'missing\s+temperature\s+logs',
                r'(documentation|compliance)\s+issues',
                r'had\s+.+\s+issues',
                r'violations',
                r'which\s+.+\s+had\s+missing',
                r'which\s+.+\s+had\s+documentation',
            ]
        }
        
        # Router prompt template
        self.system_prompt = """You are an expert data router for a pharmaceutical supply chain system. 
Your task is to analyze queries and determine which data sources are most relevant.

Available data sources:
1. inventory - Contains product data including stock levels, expiry dates, reorder points, and item details
2. transport - Contains shipping information, delivery status, delays, and logistics data
3. guidelines - Contains regulatory requirements, government guidelines, and compliance information
4. policy - Contains company policies, standard operating procedures, and internal rules

For each query, determine which sources are most relevant for answering the question.
Some queries may require multiple data sources to provide a complete answer."""

        self.router_template = """
Query: {query}

Determine which data sources are most relevant for answering this query.
Return a JSON array of source names in order of relevance (most relevant first).
Only include sources that are directly relevant to the query.

Example format: ["inventory", "transport"]"""

        logger.info(f"Initialized DataRouter with {len(self.data_sources)} sources")
    
    def determine_data_sources(self, query: str, entities: Optional[Dict[str, List[str]]] = None, 
                               query_intent: Optional[str] = None) -> Dict[str, Any]:
        """
        Determine relevant data sources for a query, supporting multi-source queries.
        
        Args:
            query (str): The user's query
            entities (Dict): Optional detected entities from QueryEnhancer
            query_intent (str): Optional query intent classification
            
        Returns:
            Dict: Dictionary with sources and confidence scores
        """
        try:
            logger.info(f"Determining data sources for query: '{query}'")
            query_lower = query.lower()
            
            # Use rule-based approach first (faster response)
            routing_result = self._rule_based_source_determination(query_lower, entities, query_intent)
            
            # Only use LLM if rule-based approach had low confidence
            if routing_result['confidence'] < 0.7:
                logger.info("Rule-based routing had low confidence, using LLM")
                llm_result = self._llm_based_source_determination(query)
                
                # Merge results with preference to high-confidence rule-based decisions
                final_sources = []
                for source in routing_result['sources']:
                    if source not in final_sources:
                        final_sources.append(source)
                
                for source in llm_result:
                    if source not in final_sources:
                        final_sources.append(source)
                
                routing_result['sources'] = final_sources
                routing_result['confidence'] = max(routing_result['confidence'], 0.8)  # LLM boosts confidence
            
            # If still no sources found, default to most common sources
            if not routing_result['sources']:
                logger.warning("No sources determined, using defaults")
                routing_result['sources'] = ["inventory", "transport"]
                routing_result['confidence'] = 0.5
            
            logger.info(f"Determined sources: {routing_result['sources']} with confidence {routing_result['confidence']}")
            return routing_result
            
        except Exception as e:
            logger.error(f"Error determining data sources: {str(e)}")
            logger.error(traceback.format_exc())
            # Default to most common sources in case of error
            return {
                'sources': ["inventory", "transport"],
                'confidence': 0.3,
                'agent_type': 'vector_rag',
                'query_type': 'factual'
            }
    
    def _rule_based_source_determination(self, query_lower: str, 
                                        entities: Optional[Dict[str, List[str]]] = None,
                                        query_intent: Optional[str] = None) -> Dict[str, Any]:
        """Use keyword matching and pattern recognition to determine relevant sources with confidence scores."""
        source_scores = {source: 0.0 for source in self.data_sources.keys()}
        
        # Check query patterns to determine agent type
        agent_type = self._determine_agent_type(query_lower)
        
        # Tag the query as factual or analytical
        query_type = query_intent if query_intent else ('analytical' if agent_type == 'analytics' else 'factual')
        if agent_type == 'hybrid':
            query_type = 'hybrid'
        
        logger.info(f"Query type determined as: {query_type}")
        logger.info(f"Agent type determined as: {agent_type}")
        
        # Use medication entity information if available
        if entities and entities.get('medications'):
            for med in entities['medications']:
                med_sources = self.medication_sources.get(med.lower(), [])
                for source in med_sources:
                    source_scores[source] += 0.4  # Significant boost for medication-specific sources
                    
        # Use regulatory entity information if available
        if entities and entities.get('regulatory'):
            for term in entities['regulatory']:
                if term.lower() in ['fda', 'phmsa', 'tsa', 'un 3373', 'triple packaging']:
                    source_scores['guidelines'] += 0.5  # Strong signal for guidelines
                elif term.lower() in ['temperature log', 'excursion']:
                    source_scores['transport'] += 0.4  # Strong signal for transport
                    
        # Use classification entity information
        if entities and entities.get('classification'):
            for term in entities['classification']:
                if term.lower() in ['class a', 'class b', 'class c']:
                    source_scores['policy'] += 0.4  # Strong signal for policy
                    source_scores['inventory'] += 0.3  # Also relevant for inventory
                elif term.lower() in ['critical stock', 'high demand', 'stockout']:
                    source_scores['inventory'] += 0.5  # Strong signal for inventory
        
        # If we have a hybrid query that mentions logs, documentation, or compliance issues
        hybrid_patterns = [
            r'missing\s+.+\s+logs', 
            r'documentation\s+issues',
            r'compliance\s+issues',
            r'violations', 
            r'had\s+.+\s+issues'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in hybrid_patterns):
            source_scores['guidelines'] += 0.4
            source_scores['transport'] += 0.4
        
        # Check for explicit web search requests
        explicit_web_search_patterns = [
            r'search\s+for',
            r'find\s+online',
            r'look\s+up',
            r'recent',
            r'latest',
            r'news',
            r'current',
            r'update'
        ]
        
        for pattern in explicit_web_search_patterns:
            if re.search(pattern, query_lower):
                source_scores['web'] += 0.3
        
        # Score based on keyword matching
        for source, keywords in self.source_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    source_scores[source] += 0.15
                    
                    # Add bonus for exact phrase matches
                    if f" {keyword} " in f" {query_lower} ":
                        source_scores[source] += 0.1
        
        # Get max score and normalize others
        max_score = max(source_scores.values())
        min_threshold = 0.15  # Minimum score to be considered relevant
        
        # For hybrid queries requiring guidelines + data analysis
        if agent_type == 'hybrid':
            logger.info("Detected hybrid query requiring multiple agents")
            
            # Ensure we have at least guidelines and one data source
            if source_scores['guidelines'] < min_threshold:
                source_scores['guidelines'] = min_threshold + 0.1
                
            # Make sure we have at least one data source (inventory or transport)
            if source_scores['inventory'] < min_threshold and source_scores['transport'] < min_threshold:
                # Choose the higher scored one
                if source_scores['inventory'] >= source_scores['transport']:
                    source_scores['inventory'] = min_threshold + 0.05
                else:
                    source_scores['transport'] = min_threshold + 0.05
        
        # Filter to sources above threshold and sort by score
        relevant_sources = []
        for source, score in sorted(source_scores.items(), key=lambda x: x[1], reverse=True):
            if score >= min_threshold or (source == 'web' and score > 0):
                relevant_sources.append(source)
        
        # Calculate overall confidence based on score distribution
        confidence = min(1.0, max_score * 1.5)  # Scale up but cap at 1.0
        
        # If we have exactly one high-scoring source, increase confidence
        if max_score > 0.5 and len([s for s, v in source_scores.items() if v > 0.3]) == 1:
            confidence = min(1.0, confidence + 0.2)
        
        return {
            'sources': relevant_sources[:3],  # Limit to top 3 sources
            'confidence': confidence,
            'scores': source_scores,
            'agent_type': agent_type,
            'query_type': query_type
        }
    
    def _determine_agent_type(self, query_lower: str) -> str:
        """Determine which agent should handle this query."""
        # Check for hybrid patterns first (most specific)
        for pattern in self.query_to_agent_mapping['hybrid']:
            if re.search(pattern, query_lower):
                return 'hybrid'
        
        # Check analytics patterns
        for pattern in self.query_to_agent_mapping['analytics']:
            if re.search(pattern, query_lower):
                return 'analytics'
        
        # Check vector RAG patterns
        for pattern in self.query_to_agent_mapping['vector_rag']:
            if re.search(pattern, query_lower):
                return 'vector_rag'
        
        # Default based on simple heuristics
        if any(term in query_lower for term in ['how many', 'trend', 'compare', 'list all', 'current stock']):
            return 'analytics'
        
        # Default to vector RAG for most queries
        return 'vector_rag'
    
    def _llm_based_source_determination(self, query: str) -> List[str]:
        """Use LLM to determine relevant sources when rules are not confident."""
        try:
            logger.info(f"Using LLM for source determination: '{query}'")
            
            response = self.client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.router_template.format(query=query)}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info(f"LLM routing response: {result_text}")
            
            # Try to parse the result as JSON
            try:
                import json
                sources = json.loads(result_text)
                
                # Make sure it's a list of strings
                if isinstance(sources, list) and all(isinstance(s, str) for s in sources):
                    # Validate source names
                    sources = [s for s in sources if s in self.data_sources]
                    return sources
                else:
                    logger.warning(f"LLM returned invalid source format: {result_text}")
                    return self._extract_sources_from_text(result_text)
            except Exception as e:
                logger.warning(f"Failed to parse LLM result as JSON: {str(e)}")
                return self._extract_sources_from_text(result_text)
            
        except Exception as e:
            logger.error(f"Error in LLM source determination: {str(e)}")
            logger.error(traceback.format_exc())
            return ["inventory", "transport"]  # Default sources if LLM fails
    
    def _extract_sources_from_text(self, text: str) -> List[str]:
        """Extract source names from text when JSON parsing fails."""
        sources = []
        valid_sources = list(self.data_sources.keys())
        
        for source in valid_sources:
            if source.lower() in text.lower():
                sources.append(source)
        
        return sources if sources else ["inventory"]
    
    def route_query(self, query: str, enhancement_result: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Route a query to the appropriate data sources and load relevant data.
        
        Args:
            query (str): The enhanced query to route
            enhancement_result (Dict): Optional result from QueryEnhancer
            
        Returns:
            Dict[str, Any]: Dictionary with data from relevant sources
        """
        try:
            logger.info(f"Routing query: '{query}'")
            
            # Extract entities and intent if available from enhancement
            entities = None
            query_intent = None
            
            if enhancement_result:
                entities = enhancement_result.get('entities')
                query_intent = enhancement_result.get('query_intent')
                
                if entities:
                    logger.info(f"Using entities from QueryEnhancer: {entities}")
                if query_intent:
                    logger.info(f"Using query intent from QueryEnhancer: {query_intent}")
            
            # Determine relevant data sources
            routing_result = self.determine_data_sources(query, entities, query_intent)
            data_sources = routing_result['sources']
            agent_type = routing_result.get('agent_type', 'vector_rag')
            query_type = routing_result.get('query_type', 'factual')
            
            # Store metadata for reference in other components
            self.agent_type = agent_type
            self.query_type = query_type
            
            # Prepare result dictionary
            result = {
                'status': 'success',
                'sources': data_sources,
                'data': {},
                'query_type': query_type,
                'agent_type': agent_type,
                'confidence': routing_result.get('confidence', 0.5)
            }
            
            # Check for web search request
            is_web_search = enhancement_result and enhancement_result.get('is_web_search', False)
            
            # Only add web search source if explicitly required by the query enhancer
            if is_web_search and 'web' not in data_sources:
                data_sources.append('web')
                logger.info("Added web search to data sources based on query patterns")
            # Remove web search if it was added by the router but not needed by the query enhancer
            elif 'web' in data_sources and not is_web_search:
                data_sources.remove('web')
                logger.info("Removed web search from data sources as not explicitly required")
            
            # Set flag for ChromaDB usage
            used_chroma_db = False
            
            # Handle each data source
            for source in data_sources:
                if source == 'web':
                    # Generate web context
                    web_context = self.web_search_agent.generate_web_context(query)
                    if web_context:
                        result['web_context'] = web_context
                    continue
                
                # For non-web sources, load data
                try:
                    # First try to use ChromaDB if available
                    if self.config.use_chroma_db and self.chroma_parser:
                        logger.info(f"Using ChromaDB for source: {source}")
                        # Map source to collection
                        collection_map = {
                            'policy': 'policies',
                            'guidelines': 'guidelines',
                            'inventory': 'inventory',
                            'transport': 'transport'
                        }
                        collection_name = collection_map.get(source, source)
                        
                        # Get data from ChromaDB
                        df = self.search_collection_as_dataframe(
                            collection_name, query, top_k=10
                        )
                        
                        if not df.empty:
                            used_chroma_db = True
                            logger.info(f"Loaded {len(df)} documents from ChromaDB collection {collection_name}")
                            
                            # For transport data, enhance with additional information
                            if source == 'transport':
                                df = self._enrich_transport_data(df)
                            
                            result['data'][source] = df
                        else:
                            logger.warning(f"No data found in ChromaDB for {source}")
                    
                    # Fallback to regular data loading if ChromaDB failed or is not enabled
                    if source not in result['data'] or result['data'][source].empty:
                        logger.info(f"Using embedded data for source: {source}")
                        df = self.load_data(source)
                        
                        if df is not None and not df.empty:
                            logger.info(f"Loaded {len(df)} rows from {source}")
                            result['data'][source] = df
                        else:
                            logger.warning(f"No data loaded for source: {source}")
                
                except Exception as e:
                    logger.error(f"Error loading data for source {source}: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # Add transport knowledge if transport is one of the sources
            if 'transport' in data_sources:
                try:
                    if hasattr(self.transport_kb, 'get_relevant_knowledge'):
                        transport_knowledge = self.transport_kb.get_relevant_knowledge(query)
                        if transport_knowledge:
                            result['transport_knowledge'] = transport_knowledge
                    else:
                        # Fallback if method doesn't exist
                        logger.warning("TransportKnowledgeBase doesn't have get_relevant_knowledge method, using fallback")
                        # Use alternative method if available or create basic knowledge
                        if hasattr(self.transport_kb, 'get_medicine_info') and 'flu vaccine' in query.lower():
                            medicine_info = self.transport_kb.get_medicine_info('flu vaccine')
                            if medicine_info:
                                result['transport_knowledge'] = {
                                    'requirements': 'Temperature-sensitive vaccines require continuous cold chain monitoring',
                                    'regulations': 'Follow FDA and PHMSA guidelines for biological substances',
                                    'medicine_info': medicine_info
                                }
                except Exception as e:
                    logger.error(f"Error getting transport knowledge: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # Record ChromaDB usage
            result['used_chroma_db'] = used_chroma_db
            
            return result
            
        except Exception as e:
            logger.error(f"Error routing query: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f"Error routing query: {str(e)}",
                'sources': ["inventory"],
                'data': {},
                'query_type': 'factual',
                'agent_type': 'vector_rag'
            }

    def search_collection_as_dataframe(self, collection_name: str, query: str, top_k: int = 10) -> pd.DataFrame:
        """Search a ChromaDB collection and return results as DataFrame."""
        try:
            results = self.chroma_parser.search_collection(collection_name, query, top_k=top_k)
            
            if not results:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Add source column if missing
            if 'source' not in df.columns:
                source_name = self.chroma_parser.collection_to_source.get(collection_name, collection_name)
                df['source'] = source_name
                
            return df
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {str(e)}")
            return pd.DataFrame()
    
    def load_data(self, source: str) -> Optional[pd.DataFrame]:
        """
        Load data from the specified source
        
        Args:
            source (str): Source identifier
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with source data or None if error
        """
        try:
            if source not in self.data_sources:
                logger.error(f"Unknown data source: {source}")
                return None
            
            if source == 'web':
                logger.info("Skipping data loading for web search")
                return None
            
            # First try to load from ChromaParser
            try:
                # Map source name to ChromaParser collection name
                collection_mapping = {
                    'inventory': 'inventory',
                    'transport': 'transport',
                    'guidelines': 'guidelines',
                    'policy': 'policies'
                }
                
                collection_name = collection_mapping.get(source)
                if collection_name and self.chroma_parser:
                    # Get dataframe from ChromaParser
                    df = self.chroma_parser.get_dataframe(collection_name)
                    
                    if not df.empty:
                        logger.info(f"Loaded {len(df)} items from ChromaParser collection {collection_name}")
                        return df
                    else:
                        logger.warning(f"No data found in ChromaParser collection {collection_name}")
            except Exception as e:
                logger.warning(f"Error loading from ChromaParser, falling back to CSV: {str(e)}")
                
            # Fall back to loading from CSV if ChromaParser fails
            file_path = self.data_sources[source]
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                return None
                
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Convert embedding column to numpy arrays
            if 'embedding' in df.columns:
                df['embedding_array'] = df['embedding'].apply(
                    lambda x: np.array([float(i) for i in x.split(',')])
                )
            
            logger.info(f"Loaded {len(df)} rows from {source}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {source}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def get_source_path(self, source: str) -> Optional[str]:
        """
        Get the file path for a given data source.
        
        Args:
            source (str): The name of the data source
            
        Returns:
            Optional[str]: The file path for the data source, or None if not found
        """
        return self.data_sources.get(source)
    
    def validate_source(self, source: str) -> bool:
        """
        Validate if a data source exists.
        
        Args:
            source (str): The name of the data source to validate
            
        Returns:
            bool: True if the source exists, False otherwise
        """
        return source in self.data_sources
    
    def _validate_embedded_data(self, df: pd.DataFrame, source: str) -> bool:
        """Validate that the embedded data has the required columns."""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in {source} data: {missing_columns}")
            return False
        return True
    
    def _process_embeddings(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Process embeddings from the dataframe."""
        try:
            # Convert string embeddings to numpy arrays
            embeddings = df['embedding'].apply(eval).values
            return np.vstack(embeddings)
        except Exception as e:
            logger.error(f"Error processing embeddings: {str(e)}")
            return None

    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        return list(self.data_sources.keys()) 

    def _enrich_transport_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich transport data with knowledge base information."""
        try:
            # Check if the dataframe has the needed columns
            if 'GenericName' not in df.columns:
                logger.warning("GenericName column not found in transport data, attempting to find alternative")
                # Try to find an alternative column or create one
                if 'metadata' in df.columns and isinstance(df['metadata'].iloc[0], dict):
                    # Try to extract from metadata if available
                    df['GenericName'] = df['metadata'].apply(
                        lambda x: x.get('generic_name', x.get('medicine_name', 'unknown'))
                        if isinstance(x, dict) else 'unknown'
                    )
                elif 'content' in df.columns:
                    # Try to extract medicine names from content
                    medicine_pattern = r'(Growth Hormone|Beta Interferon|Insulin|Vaccines|Paracetamol|Amoxicillin|Metformin|Lisinopril|Ciprofloxacin|Sildenafil|Atorvastatin|Alpha Interferon|Rituximab|Aspirin|Erythropoietin|Flu Vaccine|Adalimumab|COVID-19 Vaccine)'
                    
                    def extract_medicine(text):
                        if not isinstance(text, str):
                            return 'unknown'
                        matches = re.findall(medicine_pattern, text, re.IGNORECASE)
                        return matches[0] if matches else 'unknown'
                    
                    df['GenericName'] = df['content'].apply(extract_medicine)
                else:
                    # Create a placeholder column
                    logger.warning("Unable to determine medicine names, using placeholder")
                    df['GenericName'] = 'unknown'
            
            # Create columns for enriched data
            df['HistoricalDelays'] = None
            df['RecommendedCarriers'] = None
            df['RiskAssessment'] = None
            df['SustainabilityMetrics'] = None
            
            # Process each unique medicine
            for medicine in df['GenericName'].unique():
                if medicine == 'unknown':
                    continue
                    
                # Get medicine info from knowledge base
                medicine_info = self.transport_kb.get_medicine_info(medicine)
                if medicine_info:
                    # Get historical performance
                    if 'shipment_performance' in medicine_info:
                        perf = medicine_info['shipment_performance']
                        mask = df['GenericName'] == medicine
                        df.loc[mask, 'HistoricalDelays'] = perf.get('avg_arrival_delay')
                    
                    # Get recommended carriers
                    if 'routes' in medicine_info and 'carriers' in medicine_info['routes']:
                        carriers = medicine_info['routes']['carriers']
                        mask = df['GenericName'] == medicine
                        df.loc[mask, 'RecommendedCarriers'] = str(carriers)
                    
                    # Get risk assessment
                    if 'risk_assessment' in medicine_info:
                        risk = medicine_info['risk_assessment']
                        mask = df['GenericName'] == medicine
                        df.loc[mask, 'RiskAssessment'] = risk.get('category')
                    
                    # Get sustainability metrics
                    if 'environmental_impact' in medicine_info:
                        env = medicine_info['environmental_impact']
                        mask = df['GenericName'] == medicine
                        df.loc[mask, 'SustainabilityMetrics'] = str(env)
            
            return df
            
        except Exception as e:
            logger.error(f"Error enriching transport data: {str(e)}")
            logger.error(traceback.format_exc())
            return df

    def analyze_transport_query(self, query: str) -> Dict[str, Any]:
        """Analyze transport-specific query to determine focus areas."""
        analysis = {
            'focus_areas': [],
            'medicines': [],
            'metrics': [],
            'time_range': None
        }
        
        try:
            # Identify focus areas based on transport patterns
            for area, pattern in self.transport_keywords.items():
                if any(keyword in query.lower() for keyword in pattern):
                    analysis['focus_areas'].append(area)
            
            # Extract medicine names
            medicine_pattern = r'(Growth Hormone|Beta Interferon|Insulin|Vaccines|Paracetamol|Amoxicillin|Metformin|Lisinopril|Ciprofloxacin|Sildenafil|Atorvastatin|Alpha Interferon|Rituximab|Aspirin|Erythropoietin|Flu Vaccine|Adalimumab|COVID-19 Vaccine)'
            medicines = re.findall(medicine_pattern, query, re.IGNORECASE)
            if medicines:
                analysis['medicines'] = medicines
            
            # Extract metrics of interest
            metric_pattern = r'(delay|cost|carbon footprint|temperature|documentation|efficiency)'
            metrics = re.findall(metric_pattern, query.lower())
            if metrics:
                analysis['metrics'] = list(set(metrics))
            
            # Extract time range if present
            time_pattern = r'(last|past|previous)\s+(\d+)\s+(day|week|month|year)s?'
            time_match = re.search(time_pattern, query.lower())
            if time_match:
                analysis['time_range'] = {
                    'quantity': int(time_match.group(2)),
                    'unit': time_match.group(3)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing transport query: {str(e)}")
            return analysis 

    def _extract_time_constraints(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract time-related constraints from the query."""
        try:
            # Convert embedding to time-related features
            time_features = self._get_time_features(query_embedding)
            
            if time_features:
                return {
                    'start_date': time_features.get('start_date'),
                    'end_date': time_features.get('end_date'),
                    'period': time_features.get('period')
                }
            return None
        except Exception as e:
            logger.error(f"Error extracting time constraints: {str(e)}")
            return None

    def _apply_time_constraints(self, df: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
        """Apply time constraints to the DataFrame."""
        try:
            if 'start_date' in constraints and constraints['start_date']:
                df = df[df['DepartureDate'] >= constraints['start_date']]
            if 'end_date' in constraints and constraints['end_date']:
                df = df[df['DepartureDate'] <= constraints['end_date']]
            if 'period' in constraints and constraints['period']:
                # Apply period-specific filtering
                pass
            return df
        except Exception as e:
            logger.error(f"Error applying time constraints: {str(e)}")
            return df

    def _filter_relevant_data(self, df: pd.DataFrame, query_embedding: np.ndarray, source: str) -> pd.DataFrame:
        """Filter data based on relevance to the query."""
        try:
            if source == 'transport':
                # Filter transport data based on query relevance
                if 'DelayDays' in df.columns:
                    delay_relevance = self._compute_delay_relevance(query_embedding)
                    if delay_relevance > 0.5:  # High delay relevance
                        df = df[df['DelayDays'] > 0]
            elif source == 'inventory':
                # Filter inventory data based on query relevance
                if 'CurrentStock' in df.columns and 'ReorderPoint' in df.columns:
                    stock_relevance = self._compute_stock_relevance(query_embedding)
                    if stock_relevance > 0.5:  # High stock relevance
                        df = df[df['CurrentStock'] < df['ReorderPoint']]
            
            return df
        except Exception as e:
            logger.error(f"Error filtering relevant data: {str(e)}")
            return df

    def _compute_delay_relevance(self, query_embedding: np.ndarray) -> float:
        """Compute relevance score for delay-related queries."""
        try:
            # Implement delay relevance computation
            return 0.0  # Placeholder
        except Exception as e:
            logger.error(f"Error computing delay relevance: {str(e)}")
            return 0.0

    def _compute_stock_relevance(self, query_embedding: np.ndarray) -> float:
        """Compute relevance score for stock-related queries."""
        try:
            # Implement stock relevance computation
            return 0.0  # Placeholder
        except Exception as e:
            logger.error(f"Error computing stock relevance: {str(e)}")
            return 0.0 

    def _get_time_features(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract time-related features from query embedding."""
        try:
            # Initialize default time features
            time_features = {
                'start_date': None,
                'end_date': None,
                'period': None
            }
            
            # Get current date for reference
            current_date = pd.Timestamp.now()
            
            # Define time-related keywords and their corresponding periods
            time_keywords = {
                'today': pd.Timedelta(days=1),
                'week': pd.Timedelta(days=7),
                'month': pd.Timedelta(days=30),
                'quarter': pd.Timedelta(days=90),
                'year': pd.Timedelta(days=365)
            }
            
            # Calculate similarity with time-related embeddings
            time_scores = {}
            for keyword in time_keywords:
                try:
                    keyword_embedding = self._get_embedding(keyword)
                    if keyword_embedding is not None:
                        similarity = cosine_similarity([query_embedding], [keyword_embedding])[0][0]
                        time_scores[keyword] = similarity
                except Exception as e:
                    logger.warning(f"Error calculating similarity for {keyword}: {str(e)}")
                    continue
            
            if time_scores:
                # Get the most relevant time period
                most_relevant = max(time_scores.items(), key=lambda x: x[1])
                if most_relevant[1] > 0.3:  # Threshold for relevance
                    period = time_keywords[most_relevant[0]]
                    time_features.update({
                        'start_date': current_date - period,
                        'end_date': current_date,
                        'period': most_relevant[0]
                    })
            
            return time_features
            
        except Exception as e:
            logger.error(f"Error getting time features: {str(e)}")
            return None

    def _process_date_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Process date columns for the given source."""
        try:
            # Define date columns for each source
            date_columns = {
                'inventory': ['LastUpdated', 'ExpiryDate'],
                'transport': ['DepartureDate', 'EstimatedArrival', 'ActualArrival'],
                'guidelines': ['EffectiveDate', 'LastReviewDate'],
                'policy': ['EffectiveDate', 'ReviewDate']
            }
            
            # Get date columns for this source
            source_date_cols = date_columns.get(source, [])
            
            # Process each date column
            for col in source_date_cols:
                if col in df.columns:
                    try:
                        # Try multiple date formats
                        date_formats = [
                            '%Y-%m-%d',
                            '%Y-%m-%d %H:%M:%S',
                            '%d/%m/%Y',
                            '%m/%d/%Y',
                            '%Y/%m/%d'
                        ]
                        
                        for date_format in date_formats:
                            try:
                                df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                                if not df[col].isna().all():
                                    break
                            except Exception:
                                continue
                        
                        # Fill missing dates with appropriate values
                        if col.lower().endswith(('expirydate', 'duedate')):
                            df[col] = df[col].fillna(pd.Timestamp.max)
                        else:
                            df[col] = df[col].fillna(pd.Timestamp.min)
                            
                    except Exception as e:
                        logger.warning(f"Error processing date column {col}: {str(e)}")
                        continue
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing date columns: {str(e)}")
            return df

    def _map_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Map column names to standard format."""
        try:
            # Define column mappings for each source
            column_mappings = {
                'inventory': {
                    'item_id': 'ItemID',
                    'generic_name': 'GenericName',
                    'current_stock': 'CurrentStock',
                    'reorder_point': 'ReorderPoint',
                    'lead_time': 'LeadTimeDays',
                    'unit_cost': 'UnitCost'
                },
                'transport': {
                    'shipment_id': 'ShipmentID',
                    'item_id': 'ItemID',
                    'status': 'Status',
                    'departure': 'DepartureDate',
                    'estimated_arrival': 'EstimatedArrival',
                    'actual_arrival': 'ActualArrival',
                    'delay_days': 'DelayDays'
                },
                'guidelines': {
                    'regulation_id': 'RegulationID',
                    'category': 'Category',
                    'requirement': 'Requirement',
                    'impact': 'Impact'
                },
                'policy': {
                    'policy_id': 'PolicyID',
                    'category': 'Category',
                    'threshold': 'Threshold',
                    'action': 'Action'
                }
            }
            
            # Get mappings for this source
            source_mappings = column_mappings.get(source, {})
            
            # Create reverse mapping to handle both cases
            reverse_mappings = {v: k for k, v in source_mappings.items()}
            
            # Rename columns if they exist
            for old_col, new_col in source_mappings.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Also try reverse mapping
            for old_col, new_col in reverse_mappings.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            return df
            
        except Exception as e:
            logger.error(f"Error mapping columns: {str(e)}")
            return df

    def _validate_and_derive_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Validate required columns and derive missing ones if possible."""
        try:
            required_cols = self.source_columns.get(source, [])
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for {source}: {missing_cols}")
                
                # Try to derive missing columns
                if source == 'transport':
                    if 'DelayDays' in missing_cols and 'EstimatedArrival' in df.columns and 'ActualArrival' in df.columns:
                        try:
                            df['DelayDays'] = (df['ActualArrival'] - df['EstimatedArrival']).dt.total_seconds() / (24 * 3600)
                            missing_cols.remove('DelayDays')
                        except Exception as e:
                            logger.warning(f"Error calculating DelayDays: {str(e)}")
                    
                    if 'Status' in missing_cols and 'DelayDays' in df.columns:
                        try:
                            df['Status'] = df['DelayDays'].apply(lambda x: 
                                'Delayed' if x > 0 else 
                                'On Time' if x == 0 else 
                                'Early' if x < 0 else 'Unknown'
                            )
                            missing_cols.remove('Status')
                        except Exception as e:
                            logger.warning(f"Error deriving Status: {str(e)}")
                
                elif source == 'inventory':
                    if 'ReorderPoint' in missing_cols and 'CurrentStock' in df.columns:
                        try:
                            # Set reorder point to 20% of current stock as a basic rule
                            df['ReorderPoint'] = df['CurrentStock'].apply(lambda x: max(int(x * 0.2), 1))
                            missing_cols.remove('ReorderPoint')
                        except Exception as e:
                            logger.warning(f"Error calculating ReorderPoint: {str(e)}")
            
            # Return empty DataFrame if critical columns are still missing
            critical_cols = {
                'inventory': ['ItemID', 'CurrentStock'],
                'transport': ['ShipmentID', 'ItemID'],
                'guidelines': ['RegulationID', 'Requirement'],
                'policy': ['PolicyID', 'Action']
            }
            
            source_critical_cols = critical_cols.get(source, [])
            if any(col not in df.columns for col in source_critical_cols):
                logger.error(f"Missing critical columns for {source}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating columns: {str(e)}")
            return df

    def _process_numeric_fields(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Process numeric fields for the given source."""
        try:
            # Define numeric fields for each source
            numeric_fields = {
                'inventory': ['CurrentStock', 'ReorderPoint', 'LeadTimeDays', 'UnitCost'],
                'transport': ['DelayDays', 'Distance', 'Cost'],
                'guidelines': ['ImpactScore', 'Priority'],
                'policy': ['Threshold', 'MinValue', 'MaxValue']
            }
            
            # Get numeric fields for this source
            source_numeric_fields = numeric_fields.get(source, [])
            
            # Process each numeric field
            for field in source_numeric_fields:
                if field in df.columns:
                    try:
                        df[field] = pd.to_numeric(df[field], errors='coerce')
                        
                        # Fill missing values with appropriate defaults
                        if field in ['CurrentStock', 'ReorderPoint', 'LeadTimeDays']:
                            df[field] = df[field].fillna(0)
                        elif field in ['UnitCost', 'Cost']:
                            df[field] = df[field].fillna(0)
                        elif field in ['DelayDays']:
                            df[field] = df[field].fillna(0)
                        else:
                            df[field] = df[field].fillna(df[field].mean())
                            
                    except Exception as e:
                        logger.warning(f"Error processing numeric field {field}: {str(e)}")
                        continue
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing numeric fields: {str(e)}")
            return df

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return None

    def _prepare_response_sources(self, sources, data_dict, web_context):
        """Format sources for inclusion in the response."""
        source_info = []
        
        # Only include web search if web_context exists and has valid content
        if 'web' in sources and web_context and isinstance(web_context, list) and len(web_context) > 0:
            web_sources = []
            for idx, result in enumerate(web_context):
                if 'title' in result and 'snippet' in result:
                    web_sources.append(f"{result['title']}")
            
            if web_sources:
                source_info.append("Sources:")
                source_info.extend([f"• {source}" for source in web_sources])
        
        # Add other sources
        if 'inventory' in sources and 'inventory' in data_dict:
            inventory_sources = []
            if isinstance(data_dict['inventory'], pd.DataFrame) and not data_dict['inventory'].empty:
                for idx, row in data_dict['inventory'].head(5).iterrows():
                    if 'ProductName' in row:
                        inventory_sources.append(f"[Inventory {idx+1}] {row['ProductName']}")
                    elif 'ItemID' in row:
                        inventory_sources.append(f"[Inventory {idx+1}] {row['ItemID']}")
            
            if inventory_sources:
                source_info.append("Inventory:")
                source_info.extend([f"• {source}" for source in inventory_sources])
        
        if 'transport' in sources and 'transport' in data_dict:
            transport_sources = []
            if isinstance(data_dict['transport'], pd.DataFrame) and not data_dict['transport'].empty:
                for idx, row in data_dict['transport'].head(5).iterrows():
                    if 'ShipmentID' in row:
                        transport_sources.append(f"[Transport {idx+1}] {row['ShipmentID']}")
            
            if transport_sources:
                source_info.append("Transport:")
                source_info.extend([f"• {source}" for source in transport_sources])
        
        if 'guidelines' in sources and 'guidelines' in data_dict:
            guidelines_sources = []
            if isinstance(data_dict['guidelines'], pd.DataFrame) and not data_dict['guidelines'].empty:
                for idx, row in data_dict['guidelines'].head(5).iterrows():
                    if 'Title' in row:
                        guidelines_sources.append(f"[Guidelines {idx+1}] {row['Title']}")
                    else:
                        guidelines_sources.append(f"[Guidelines {idx+1}] Regulatory document {idx+1}")
            
            if guidelines_sources:
                source_info.append("Guidelines:")
                source_info.extend([f"• {source}" for source in guidelines_sources])
        
        return "\n".join(source_info) 