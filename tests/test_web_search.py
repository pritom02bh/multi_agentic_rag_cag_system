import sys
import os
import logging
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.web_search import WebSearchAgent
from config.settings import AppConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_web_search():
    """Test that the web search agent works correctly."""
    try:
        # Initialize agent
        agent = WebSearchAgent()
        logger.info("WebSearchAgent initialized successfully")
        
        # Test search function with a simple query
        query = "latest pharmaceutical regulations FDA 2023"
        search_results = agent.search(query)
        
        logger.info(f"Search status: {search_results['status']}")
        
        if search_results['status'] == 'success':
            logger.info(f"Found {len(search_results['results'])} results")
            # Print first result
            if search_results['results']:
                first_result = search_results['results'][0]
                logger.info(f"First result: {first_result.get('title', 'No title')}")
                logger.info(f"Snippet: {first_result.get('snippet', 'No snippet')[:100]}...")
        else:
            logger.error(f"Search failed: {search_results['message']}")
            
        # Test context generation
        context = agent.generate_web_context(query)
        logger.info(f"Generated context length: {len(context)}")
        logger.info(f"Context preview: {context[:200]}...")
        
        return search_results['status'] == 'success'
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting WebSearchAgent test")
    success = test_web_search()
    
    if success:
        logger.info("Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("Test failed")
        sys.exit(1) 