import sys
import os
import logging
import argparse
from dotenv import load_dotenv
import subprocess
from app import app
from agents.agent_registry import init_agents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run the pharmaceutical supply chain management system')
    parser.add_argument('--host', type=str, default=None, help='Host address to bind to')
    parser.add_argument('--port', type=int, default=None, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--install-deps', action='store_true', help='Install missing dependencies automatically')
    return parser.parse_args()

def install_dependencies(deps):
    """Install missing dependencies."""
    logger.info(f"Installing missing dependencies: {', '.join(deps)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + deps)
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_deps = []
    
    # Check for transformers
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    # Check for bitsandbytes (needed for quantization)
    try:
        import bitsandbytes
    except ImportError:
        missing_deps.append("bitsandbytes")
    
    return missing_deps

def init_environment():
    """Initialize the environment variables."""
    load_dotenv()
    
    # Check for required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file or environment.")
        return False
    
    return True

def main():
    """Main entry point for the application."""
    try:
        # Parse command-line arguments
        args = parse_args()
        
        # Check for missing dependencies
        missing_deps = check_dependencies()
        if missing_deps and args.install_deps:
            if not install_dependencies(missing_deps):
                logger.warning("Will continue without installing missing dependencies")
        elif missing_deps:
            logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
            logger.warning("The application may not function correctly without them.")
            logger.warning("Run with --install-deps to install them automatically.")
        
        # Initialize environment variables
        if not init_environment():
            logger.error("Failed to initialize environment. Please check your configuration.")
            sys.exit(1)
        
        # Initialize agents
        if not init_agents():
            logger.error("Failed to initialize agents. Please check your configuration.")
            sys.exit(1)
        
        # Get configuration from environment or command-line arguments
        host = args.host or os.getenv('HOST', '0.0.0.0')
        port = args.port or int(os.getenv('PORT', 5000))
        debug = args.debug or os.getenv('DEBUG', 'True').lower() == 'true'
        
        # Run the application
        logger.info(f"Starting application on {host}:{port} (Debug: {debug})")
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug
        )
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 