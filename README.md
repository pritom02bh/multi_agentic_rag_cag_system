# Multi-Agent RAG System for Pharmaceutical Supply Chain Management

This system implements a multi-agent architecture for pharmaceutical supply chain management, leveraging Retrieval Augmented Generation (RAG) and advanced analytics capabilities to provide comprehensive responses to user queries.

## System Architecture

The system consists of the following components:

### Core Components

1. **Router Agent**: Analyzes user queries and routes them to the appropriate specialized agent.
2. **RAG Agent**: Retrieves information from the Pinecone vector database based on user queries.
3. **Web News Search Agent**: Searches for real-time information from the web and news sources.
4. **Analytics Agent**: Analyzes numerical data and generates visualizations.
5. **CAG (Cache Augmented Generation) Agent**: Provides policy and transport information.
6. **Aggregator Agent**: Combines responses from multiple agents into a coherent output.

### Web Interface

The system includes a Flask-based web interface for interacting with the agents:
- Modern, responsive UI
- Interactive pipeline visualization
- Support for complex queries and visualization display

## System Requirements

- Python 3.8+
- Pinecone account (for vector database)
- OpenAI API key
- News API key (optional, for web search functionality)
- Redis (optional, for rate limiting and caching)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/multi-agentic-rag-v2.git
   cd multi-agentic-rag-v2
   ```

2. **Set up virtual environment**:
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your API keys and configuration

## Configuration

### Required Environment Variables

The system requires several environment variables to be set in the `.env` file:

#### API Keys
- `OPENAI_API_KEY`: Your OpenAI API key for LLM access
- `PINECONE_API_KEY`: Your Pinecone API key for vector database access
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., us-east1-gcp)
- `PINECONE_INDEX`: The name of your Pinecone index (default: medical-supply-chain)

#### Optional Keys
- `NEWS_API_KEY`: For the Web Search Agent (if not provided, web search will be disabled)
- Redis configuration (if not provided, in-memory storage will be used)

See `.env.example` for all configurable options.

### Setting Up Pinecone

1. Create a Pinecone account at [pinecone.io](https://www.pinecone.io/)
2. Create a new index with the following settings:
   - Name: medical-supply-chain
   - Dimension: 384
   - Metric: cosine
3. Copy your API key and environment to the `.env` file

## Running the Application

### Development Mode

```bash
python run.py
```

The application will be available at http://localhost:5000

### Using the Batch Script (Windows)

```bash
start.bat
```

## Usage Examples

The system can answer various questions about pharmaceutical supply chain:

- **Inventory Queries**: "What is the current inventory status?"
- **Transport Queries**: "Show me the status of recent shipments"
- **Policy Queries**: "What are our policies on handling expired medications?"
- **Analytics Queries**: "Analyze our inventory turnover rate"
- **Comprehensive Queries**: "Give me a complete overview of our cold chain logistics"

## Advanced Configuration

### Customizing Agent Behavior

You can modify agent configurations in the `.env` file:
- Adjust the `OPENAI_MODEL` to use different models (gpt-3.5-turbo, gpt-4, etc.)
- Configure RAG parameters like `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K`

### Adding Custom Data

The system uses Pinecone to store vectorized data across multiple namespaces:
- inventory: Product information and stock levels
- transport: Shipping and logistics data
- policies: Company guidelines and procedures

To add custom data, use the data ingestion scripts in the data directory.

## Troubleshooting

### Common Issues

1. **API Rate Limits**: If you encounter rate limit errors, consider adjusting the `BATCH_SIZE` in your `.env` file.

2. **Connection Issues**: Ensure your Pinecone and OpenAI API keys are correctly set in the `.env` file.

3. **Memory Issues**: If the application crashes due to memory constraints, try:
   - Reducing `TOP_K` value
   - Lowering the `CHUNK_SIZE`
   - Using a more efficient model

4. **Visualization Errors**: If visualizations don't display correctly, check browser console for errors and ensure you have proper permissions for accessing the data.

## License

This project is licensed under the MIT License - see the LICENSE file for details.




I'll help you understand how to run the system so that data will be used by the CAG agent. Let me check the project structure first to find the main entry point.

Ran tool

Read file: run.py

Read file: .env

```magicpython:agents/agent_registry.py
        # Create agents
        from .aggregator_agent import AggregatorAgent
        from .augmented_agents import RAGAgent, WebSearchAgent, EnhancedAnalyticsAgent, CAGAgent
        from .router_agent import RouterAgent
        
        # Initialize router agent first
        router_agent = RouterAgent(router_config)
        registry.register_agent(AgentType.ROUTER, router_agent)
        
        # Initialize aggregator agent
        aggregator_agent = AggregatorAgent(openai_api_key=openai_api_key)
        
        # Initialize RAG agent for inventory data
        rag_agent = RAGAgent(
            rag_config,
            pinecone_client,
            pinecone_index,
            namespaces=["inventory"]  # Only use inventory namespace for RAG
        )
        registry.register_agent(AgentType.RAG, rag_agent)
        
        # Initialize web search agent
        web_agent = None
        if news_api_key:
            web_agent = WebSearchAgent(web_search_config, news_api_key)
            registry.register_agent(AgentType.WEB_SEARCH, web_agent)
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
        registry.register_agent(AgentType.ENHANCED_ANALYTICS, analytics_agent)
        logger.info("Analytics agent initialized and registered")
        
        # Initialize CAG agent with sub-agents
        cag_config = AgentConfig(
            system_prompt="You are a collaborative agent specialized in policy, transport, and guidelines for pharmaceutical supply chain management. You coordinate multiple specialized sub-agents to provide comprehensive answers.",
            model_name="gpt-4",
            temperature=0.3,
            max_tokens=1500,
            openai_api_key=openai_api_key
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
        registry.register_agent(AgentType.CAG, cag_agent)
        logger.info("CAG agent initialized and registered with sub-agents")
        
        logger.info("All agents initialized and registered successfully")
```

```magicpython:agents/aggregator_agent.py
            # Set source components
            rag_response = agent_responses.get("rag", {})
            if isinstance(rag_response, str):
                rag_response = {"response": rag_response}
                
            cag_response = agent_responses.get("cag", {})
            if isinstance(cag_response, str):
                cag_response = {"response": cag_response}
                
            web_response = agent_responses.get("web_search", {})
            if isinstance(web_response, str):
                web_response = {"response": web_response}
                
            analytics_response = agent_responses.get("analytics", {})
            if isinstance(analytics_response, str):
                analytics_response = {"response": analytics_response}
            
            # Extract visualizations if available
            if "visualization" in rag_response:
                combined_response["visualization"] = rag_response["visualization"]
            elif "visualization" in analytics_response:
                combined_response["visualization"] = analytics_response["visualization"]
            
            # Process based on query type
            query_type = routing_info.get("query_type", "general") if routing_info else "general"
            
            # For inventory queries (RAG-based)
            if query_type == "inventory":
                # Use RAG response as primary
                if rag_response:
                    combined_response["summary"] = rag_response.get("response", "")
                    combined_response["details"] = self._extract_section(rag_response.get("response", ""), "details")
                    combined_response["metadata"]["sources"].append({
                        "type": "inventory",
                        "source": "Inventory Database",
                        "confidence": 0.95
                    })
                    
                    # Format inventory data as a table if available
                    if "data" in rag_response:
                        combined_response["details"] += "\n\n" + self._format_structured_data(
                            rag_response["data"], "inventory"
                        )
            
            # For transport queries (CAG-based)
            elif query_type == "transport":
                if cag_response:
                    combined_response["summary"] = cag_response.get("response", "")
                    combined_response["metadata"]["sources"].append({
                        "type": "transport",
                        "source": "Transport History",
                        "confidence": 0.9
                    })
                    
                    # Add any relevant data from CAG response
                    if "data" in cag_response:
                        combined_response["details"] = self._format_structured_data(
                            cag_response["data"], "transport"
                        )
            
            # For policy queries (CAG-based)
            elif query_type == "policy":
                if cag_response:
                    combined_response["summary"] = cag_response.get("response", "")
                    combined_response["metadata"]["sources"].append({
                        "type": "policy",
                        "source": "Inventory Management Policy",
                        "confidence": 0.95
                    })
                    
                    # Extract policy details if available
                    policy_details = self._extract_section(cag_response.get("response", ""), "details")
                    if policy_details:
                        combined_response["details"] = policy_details
            
            # For guidelines queries (CAG-based)
            elif query_type == "guidelines":
                if cag_response:
                    combined_response["summary"] = cag_response.get("response", "")
                    combined_response["metadata"]["sources"].append({
                        "type": "guidelines",
                        "source": "US Government Guidelines",
                        "confidence": 0.95
                    })
                    
                    # Extract guideline details if available
                    guideline_details = self._extract_section(cag_response.get("response", ""), "details")
                    if guideline_details:
                        combined_response["details"] = guideline_details
```
Now that we've updated the code to properly include and utilize the CAG agent, here's how to run the system so that data will be used by the CAG agent:

### Step 1: Make sure your environment is set up correctly

1. Check your `.env` file (which you already have) and ensure it contains:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_ENVIRONMENT`
   - `PINECONE_INDEX`
   - `NEWS_API_KEY` (optional but recommended for web search capability)

2. Ensure you have the necessary dependencies installed:
   ```
   pip install -r requirements.txt
   ```

### Step 2: Run the application

Run the application using the `run.py` script:

```
python run.py
```

This will:
1. Load environment variables
2. Initialize the agents including the CAG agent
3. Start the Flask server

### Step 3: Send requests that will utilize the CAG agent

The CAG agent will automatically be used for:
- Transport queries
- Policy queries 
- Guidelines queries

Here are some example queries you could use that will trigger CAG:

1. **Transport queries**:
   - "What are our current transport options for cold chain items?"
   - "Explain the route optimization for medication delivery to rural areas"
   - "What are the shipping requirements for controlled substances?"

2. **Policy queries**:
   - "What is our policy on medication expiration dates?"
   - "Explain our inventory management policy for high-value drugs"
   - "What guidelines do we follow for reordering critical medications?"

3. **Guidelines queries**:
   - "What are the FDA guidelines for storing insulin?"
   - "Explain the regulatory requirements for transporting vaccines"
   - "What government regulations apply to controlled substance inventory?"

### Step 4: Verify CAG is being used

You can verify that the CAG agent is being used by:

1. **Checking the logs**:
   - Look for log messages that say "Routing query to CAG agent"
   - Look for log messages like "CAG agent processing query"

2. **Examining the response**:
   - The response should have a "sources" section that shows data came from multiple sources
   - For transport, policy, and guidelines queries, you should see appropriate source references

### Step 5: Test with various query types to see different routing

Send different types of queries to see how the router directs them:

1. **Inventory queries** (will go to RAG):
   - "What's our current stock of paracetamol?"
   - "How many ventilators do we have in inventory?"

2. **Transport/Policy/Guidelines queries** (will go to CAG):
   - "What's our cold chain protocol for vaccines?"
   - "What are the FDA guidelines for storing antibiotics?"

3. **Analytics queries** (will go to EnhancedAnalyticsAgent):
   - "Analyze the trend of antibiotic usage over the last 6 months"
   - "Visualize our inventory distribution by medication category"

4. **News/real-time queries** (will go to WebSearchAgent):
   - "Any recent FDA announcements about drug recalls?"
   - "Latest news on pharmaceutical supply chain disruptions"

The system you now have will use CAG for non-inventory focused queries, particularly those related to transport logistics, policies, and regulatory guidelines. The CAG will coordinate responses from multiple agents to provide comprehensive answers.
