import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo-16k')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    
    # Data Sources
    DATA_SOURCES = {
        'inventory': 'embedded_data/inventory_data_embedded.csv',
        'transport': 'embedded_data/transport_history_embedded.csv',
        'guidelines': 'embedded_data/us_government_guidelines_for_medicine_transportation_and_storage_embedded.csv',
        'policy': 'embedded_data/inventory_management_policy_embedded.csv'
    }
    
    # FAISS Configuration
    FAISS_INDEX_PATH = 'indexes/'
    TOP_K_RESULTS = 19
    
    # Session Configuration
    SESSION_LIFETIME = 3600  # 1 hour
    
    # Chart Configuration
    CHART_TYPES = ['bar', 'line', 'pie', 'scatter']
    MAX_CHART_DATA_POINTS = 20
    
    # Response Configuration
    MAX_RESPONSE_LENGTH = 4096
    TEMPERATURE = 0.7
    
    # System Prompts
    QUERY_ENHANCEMENT_PROMPT = """
    Enhance the following query to make it more effective for searching pharmaceutical supply chain data:
    Query: {query}
    Enhanced Query:"""
    
    ROUTER_PROMPT = """
    Determine the most relevant data source for the following query:
    Query: {query}
    Available sources: inventory, transport, guidelines, policy
    Most relevant source:"""
    
    RAG_PROMPT = """
    Based on the following context, provide a detailed answer to the query.
    Context: {context}
    Query: {query}
    Answer:"""
    
    ANALYSIS_PROMPT = """
    Analyze the following response and extract key insights. If numerical data is present,
    suggest appropriate visualizations.
    Response: {response}
    Analysis:""" 