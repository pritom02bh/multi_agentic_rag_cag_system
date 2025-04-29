import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AppConfig:
    """Application configuration settings"""
    
    def __init__(self):
        """Initialize configuration with default values."""
        # Load environment variables
        load_dotenv()
        
        # API Configuration
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
        self.COMPLETION_MODEL = os.getenv('COMPLETION_MODEL', 'gpt-4o')  # Model for completions
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        self.SERPER_API_KEY = os.getenv('SERPER_API_KEY', '')
        self.MAX_WEB_RESULTS = int(os.getenv('MAX_WEB_RESULTS', '5'))  # Maximum number of web search results
        
        # Query Enhancement Settings
        self.MAX_QUERY_LENGTH = 500
        self.QUERY_ENHANCEMENT_TEMPLATE = """
        Please enhance the following query for pharmaceutical inventory analysis:
        Query: {query}
        
        Enhance it to include relevant pharmaceutical inventory management concepts and metrics.
        """
        
        # Data Sources
        self.DATA_SOURCES = {
            'inventory': 'embedded_data/inventory_data_embedded.csv',
            'transport': 'embedded_data/transport_history_embedded.csv',
            'guidelines': 'embedded_data/us_government_guidelines_for_medicine_transportation_and_storage_embedded.csv',
            'policy': 'embedded_data/inventory_management_policy_embedded.csv'
        }
        
        # Required Fields for Analysis
        self.REQUIRED_INVENTORY_FIELDS = [
            'ItemID',
            'GenericName',
            'CurrentStock',
            'ReorderPoint',
            'UnitCost',
            'SellingPrice',
            'ExpiryDate',
            'Status'
        ]
        
        # Analysis Configuration
        self.ANALYSIS_THRESHOLDS = {
            'low_stock': 0.2,  # 20% of reorder point
            'expiry_warning_days': 90,  # Warning for items expiring in 90 days
            'high_value_threshold': 1000  # Items worth more than $1000
        }
        
        # Response Configuration
        self.MAX_RESPONSE_LENGTH = 2000
        self.RAG_TEMPLATE = """
        Based on the following context and analysis, please provide a detailed response:
        
        Context: {context}
        Query: {query}
        
        Focus on:
        1. Current inventory status
        2. Items requiring attention
        3. Value analysis
        4. Recommendations
        """
        
        # Session Configuration
        self.SESSION_LIFETIME = 3600  # 1 hour in seconds
        
        # Chart Configuration
        self.CHART_TYPES = {
            'stock_levels': 'bar',
            'value_distribution': 'pie',
            'expiry_timeline': 'line',
            'stock_vs_reorder': 'scatter'
        }
        
        # System Prompts
        self.ROUTER_PROMPT_TEMPLATE = """
        Determine the most relevant data source for the following query:
        Query: {query}
        
        Available sources:
        - inventory: Stock levels, item details, and inventory metrics
        - transport: Shipping and logistics information
        - guidelines: Standard operating procedures and guidelines
        - policy: Company policies and compliance requirements
        
        Return only the source name (inventory, transport, guidelines, or policy).
        """

        # RAG configuration
        self.TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '5'))
        self.use_chroma_db = True  # Use ChromaDB by default
        self.chroma_parser = None  # Will be set during initialization
        
        # Path configurations
        self.DATA_DIR = 'data'
        self.VECTOR_DIR = 'vector_db'
        self.EXPORT_DIR = 'exports'
        self.CSV_DATA_DIR = 'csv_data'
        self.EMBEDDED_DATA_DIR = 'embedded_data'

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
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
    TOP_K_RESULTS = 50
    
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