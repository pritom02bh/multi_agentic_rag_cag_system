from typing import Dict, Type, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import openai
from langchain_community.tools import BaseTool
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from .augmented_tools import get_augmented_tools
from .augmented_agents import CAGAgent, RAGAgent, WebSearchAgent, EnhancedAnalyticsAgent
import logging
from threading import Lock
import os
from pinecone import Pinecone, PineconeException
from .base_agent import BaseAgent, AgentType, AgentConfig
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Registry for managing and accessing different types of agents."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, pinecone_api_key=None, pinecone_env=None, search_api_key=None, index_name="medical-supply-chain"):
        """Get or create a singleton instance of AgentRegistry."""
        if cls._instance is None:
            if not pinecone_api_key or not pinecone_env:
                # Try to get from environment
                pinecone_api_key = os.getenv("PINECONE_API_KEY")
                pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
                search_api_key = os.getenv("NEWS_API_KEY")
                index_name = os.getenv("PINECONE_INDEX", "medical-supply-chain")
                
            if not pinecone_api_key or not pinecone_env:
                raise ValueError("Pinecone API key and environment are required")
                
            cls._instance = cls(
                pinecone_api_key=pinecone_api_key,
                pinecone_env=pinecone_env,
                search_api_key=search_api_key,
                index_name=index_name
            )
        return cls._instance
    
    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_env: str,
        search_api_key: Optional[str] = None,
        index_name: str = "medical-supply-chain"
    ):
        self._agents: Dict[AgentType, BaseAgent] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(
                api_key=pinecone_api_key,
                environment=pinecone_env
            )
            
            # Store configuration
            self.pinecone_api_key = pinecone_api_key
            self.pinecone_env = pinecone_env
            self.search_api_key = search_api_key
            self.index_name = index_name
            
            # Get the index
            try:
                self.index = self.pc.Index(self.index_name)
                # Verify index exists and get stats
                stats = self.index.describe_index_stats()
                logger.info(f"Connected to Pinecone index '{index_name}' with {stats.total_vector_count} vectors")
                logger.info(f"Available namespaces: {', '.join(stats.namespaces.keys())}")
            except PineconeException as e:
                logger.error(f"Failed to connect to Pinecone index: {str(e)}")
                raise
            
            # Initialize default agents
            self._init_default_agents()
            
        except PineconeException as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AgentRegistry: {str(e)}")
            raise
    
    def _init_default_agents(self):
        """Initialize the default set of agents."""
        try:
            # Get the OpenAI model from environment
            openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            
            # Initialize RAG agent with namespace support
            self.register_agent(
                RAGAgent(
                    AgentConfig(
                        model_name=openai_model,
                        system_prompt="You are a helpful assistant specializing in medical supply chain management. You have access to information about transport, guidelines, inventory, and policies.",
                        temperature=0.7,
                        max_tokens=1000
                    ),
                    pinecone_client=self.pc,
                    index_name=self.index_name,
                    namespaces=['transport', 'guidelines', 'inventory', 'policies']
                )
            )
            logger.info("Registered RAG agent successfully")
            
            # Initialize Web Search agent if API key is available
            if self.search_api_key:
                self.register_agent(
                    WebSearchAgent(
                        AgentConfig(
                            model_name=openai_model,
                            system_prompt="You are a web search specialist focusing on medical supply chain news and updates.",
                            temperature=0.7,
                            max_tokens=1000
                        ),
                        search_api_key=self.search_api_key
                    )
                )
                logger.info("Registered Web Search agent successfully")
            
            # Initialize Enhanced Analytics agent with namespace support
            self.register_agent(
                EnhancedAnalyticsAgent(
                    AgentConfig(
                        model_name=openai_model,
                        system_prompt="You are an analytics specialist focusing on medical supply chain data visualization and analysis.",
                        temperature=0.7,
                        max_tokens=1000
                    ),
                    pinecone_client=self.pc,
                    index_name=self.index_name,
                    namespaces=['transport', 'inventory']  # Focus on transport and inventory data for analytics
                )
            )
            logger.info("Registered Enhanced Analytics agent successfully")
            
            # Initialize CAG agent as fallback
            self.register_agent(
                CAGAgent(
                    AgentConfig(
                        model_name=openai_model,
                        system_prompt="You are a conversational agent specializing in medical supply chain management.",
                        temperature=0.7,
                        max_tokens=1000
                    )
                )
            )
            logger.info("Registered CAG agent successfully")
            
        except Exception as e:
            logger.error(f"Error initializing default agents: {str(e)}")
            raise

    async def process_query(self, query: str, agent_type: AgentType) -> str:
        """Process a query using a specific agent type asynchronously."""
        try:
            agent = self.get_agent(agent_type)
            if not agent:
                raise ValueError(f"Agent type {agent_type} not found")
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                agent.process,
                query
            )
            return response
        except Exception as e:
            logger.error(f"Error processing query with {agent_type}: {str(e)}")
            raise

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent in the registry."""
        with self._lock:
            agent_type = agent.get_type()
            self._agents[agent_type] = agent
            logger.info(f"Registered agent of type: {agent_type.value}")
    
    def get_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """Get an agent by type."""
        return self._agents.get(agent_type)
    
    def list_available_agents(self) -> List[str]:
        """List all available agent types."""
        return [agent_type.value for agent_type in self._agents.keys()]
    
    def get_agent_description(self, agent_type: AgentType) -> Optional[str]:
        """Get the description of an agent."""
        agent = self._agents.get(agent_type)
        return agent.config.system_prompt if agent else None
    
    def get_agent_tools(self, agent_type: AgentType) -> List[str]:
        """Get the list of tools available to an agent."""
        agent = self._agents.get(agent_type)
        return agent.config.tools if agent and agent.config.tools else []
    
    def get_augmented_response(self, query: str, agent_types: List[AgentType]) -> Dict[str, Any]:
        """Get responses from multiple agents and combine them."""
        responses = {}
        for agent_type in agent_types:
            agent = self.get_agent(agent_type)
            if agent:
                try:
                    responses[agent_type.value] = agent.process(query)
                except Exception as e:
                    responses[agent_type.value] = f"Error: {str(e)}"
        return responses 

def init_agents():
    """Initialize all agents and register them."""
    try:
        # Load configuration
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        pinecone_index = os.getenv("PINECONE_INDEX", "medical-supply-chain")
        news_api_key = os.getenv("NEWS_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Validate API keys
        if not openai_api_key:
            logger.error("OpenAI API key not found")
            return False
        
        if not pinecone_api_key or not pinecone_env:
            logger.error("Pinecone credentials not found")
            return False
        
        # Initialize Pinecone
        try:
            from pinecone import Pinecone as PineconeClient
            pinecone_client = PineconeClient(
                api_key=pinecone_api_key,
                environment=pinecone_env
            )
            logger.info(f"Pinecone client initialized in {pinecone_env}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            return False
        
        # Create agent registry
        try:
            # Try to get existing instance
            registry = AgentRegistry.get_instance()
        except Exception as e:
            logger.warning(f"Could not get existing AgentRegistry instance: {str(e)}")
            # Create new instance
            registry = AgentRegistry(
                pinecone_api_key=pinecone_api_key,
                pinecone_env=pinecone_env,
                search_api_key=news_api_key,
                index_name=pinecone_index
            )
            AgentRegistry._instance = registry  # Set as singleton instance
            logger.info("Created new AgentRegistry instance")
        
        # Base agent configurations
        base_config = AgentConfig(
            system_prompt="You are a helpful assistant for pharmaceutical supply chain management.",
            model_name=openai_model,
            temperature=0.5,
            max_tokens=1024
        )
        
        web_search_config = AgentConfig(
            system_prompt="You are a web search agent specialized in finding and summarizing real-time information about pharmaceutical supply chain topics.",
            model_name=openai_model,
            temperature=0.3,
            max_tokens=1500
        )
        
        rag_config = AgentConfig(
            system_prompt="You are a RAG agent specialized in retrieving and synthesizing information from the pharmaceutical supply chain knowledge base. Focus on providing accurate, precise information about inventory items, stock levels, and related details.",
            model_name=openai_model,
            temperature=0.3,
            max_tokens=1500
        )
        
        analytics_config = AgentConfig(
            system_prompt="You are an analytics agent specialized in generating insights and visualizations from pharmaceutical supply chain data.",
            model_name=openai_model,
            temperature=0.3,
            max_tokens=1500
        )
        
        router_config = AgentConfig(
            system_prompt="You are a router agent that directs queries to the appropriate specialized agents based on the content. You're responsible for coordinating the flow of information between agents and ensuring that the final response meets the user's needs.",
            model_name=openai_model,
            temperature=0.3,
            max_tokens=1500
        )
        
        aggregator_config = AgentConfig(
            system_prompt="You are an aggregator agent responsible for combining information from multiple sources into coherent, useful responses. Your goal is to present information in a clear, organized manner that highlights the most important insights.",
            model_name=openai_model,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Create agents
        from .aggregator_agent import AggregatorAgent
        from .augmented_agents import RAGAgent, WebSearchAgent, EnhancedAnalyticsAgent, CAGAgent
        from .router_agent import RouterAgent
        
        # Initialize router agent first
        router_agent = RouterAgent(router_config)
        registry.register_agent(router_agent)
        
        # Initialize aggregator agent
        aggregator_agent = AggregatorAgent(
            model=openai_model,
            max_tokens=aggregator_config.max_tokens,
            openai_api_key=openai_api_key
        )
        registry.register_agent(aggregator_agent)
        
        # Initialize RAG agent for inventory data
        rag_agent = RAGAgent(
            rag_config,
            pinecone_client,
            pinecone_index,
            namespaces=["inventory"]  # Only use inventory namespace for RAG
        )
        registry.register_agent(rag_agent)
        
        # Initialize web search agent
        web_agent = None
        if news_api_key:
            web_agent = WebSearchAgent(web_search_config, news_api_key)
            registry.register_agent(web_agent)
            logger.info("Web search agent initialized and registered")
        else:
            logger.warning("News API key not provided, web search agent will not be available")
        
        # Initialize analytics agent
        analytics_agent = EnhancedAnalyticsAgent(
            analytics_config,
            pinecone_client,
            pinecone_index,
            namespaces=["inventory", "transport"]
        )
        registry.register_agent(analytics_agent)
        logger.info("Analytics agent initialized and registered")
        
        # Initialize CAG agent with sub-agents
        cag_config = AgentConfig(
            system_prompt="You are a collaborative agent specialized in policy, transport, and guidelines for pharmaceutical supply chain management. You coordinate multiple specialized sub-agents to provide comprehensive answers.",
            model_name=openai_model,
            temperature=0.3,
            max_tokens=1500
        )
        
        # Create a dictionary of sub-agents for CAG
        cag_sub_agents = {
            "rag": rag_agent,
            "enhanced_analytics": analytics_agent
        }
        
        # Add web agent if available
        if web_agent:
            cag_sub_agents["web_search"] = web_agent
            
        # Initialize and register CAG agent
        cag_agent = CAGAgent(cag_config, cag_sub_agents)
        registry.register_agent(cag_agent)
        logger.info("CAG agent initialized and registered with sub-agents")
        
        logger.info("All agents initialized and registered successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing agents: {str(e)}")
        logger.error(traceback.format_exc())
        return False 

def get_agent_registry():
    """Get the agent registry singleton instance."""
    return AgentRegistry.get_instance() 