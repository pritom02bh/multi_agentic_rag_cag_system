from flask import Flask, request, jsonify, render_template, redirect, session
from flask_cors import CORS
from flask_session import Session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restx import Api, Resource, fields
from flask_caching import Cache
from dotenv import load_dotenv
import os
import traceback
import logging
import sys
from datetime import datetime
import json
import uuid
from typing import Dict, Any
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)

# Set specific loggers to appropriate levels
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Import our components
from agents.query_enhancer import QueryEnhancer
from agents.router import DataRouter
from agents.rag import RAGAgent
from agents.analyzer import AnalyticalAgent
from agents.report_generator import ReportGenerator
from agents.web_search import WebSearchAgent
from utils.session_manager import SessionManager
from utils.chroma_manager import ChromaDBManager
from utils.chroma_parser import ChromaParser
from config.settings import AppConfig

# Load environment variables
load_dotenv(override=True)

def check_required_env_vars():
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    else:
        # Add logging to verify key format
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            logger.info(f"OpenAI API key loaded (starts with: {api_key[:5]}...)")
        else:
            logger.error("OpenAI API key is empty")

def check_required_files():
    """Check if ChromaDB collections are available and accessible"""
    try:
        # Initialize ChromaParser
        parser = ChromaParser(base_dir="vector_db_separate")
        
        # Check if collections are loaded
        collections = list(parser.collection_data.keys())
        if not collections:
            raise FileNotFoundError("No ChromaDB collections were found")
        
        # Check if each collection has documents
        empty_collections = []
        for collection in collections:
            docs = parser.get_collection_documents(collection)
            if not docs:
                empty_collections.append(collection)
        
        if empty_collections:
            logger.warning(f"The following collections have no documents: {', '.join(empty_collections)}")
        
        logger.info(f"ChromaDB collections loaded successfully: {', '.join(collections)}")
        return collections
    except Exception as e:
        raise FileNotFoundError(f"Error accessing ChromaDB: {str(e)}")

# Perform initial checks
try:
    check_required_env_vars()
    data_files = check_required_files()
    logger.info(f"All required vector collections found: {len(data_files)} collections")
except Exception as e:
    logger.error(f"Initialization check failed: {str(e)}")
    raise

# Initialize Flask app
app = Flask(__name__, 
    template_folder='ui_service/templates',
    static_folder='ui_service/static'
)

# Configure Flask app
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = 'flask_session'
app.config['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
Session(app)
CORS(app)

# Initialize cache
cache = Cache(app)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Initialize Flask-RestX
api = Api(app, version='1.0', title='Pharmaceutical RAG API',
    description='A RAG-based API for inventory management and analysis',
    doc='/api/docs',
    prefix='/api'
)

# Define namespaces
ns_chat = api.namespace('chat', description='Chat operations')
ns_system = api.namespace('system', description='System operations')

# Define models for request/response validation
chat_input = api.model('ChatInput', {
    'query': fields.String(required=True, description='User query'),
    'session_id': fields.String(required=True, description='Session identifier')
})

chat_response = api.model('ChatResponse', {
    'status': fields.String(description='Response status'),
    'response': fields.String(description='Generated natural language response'),
    'session_id': fields.String(description='Session identifier'),
    'query': fields.String(description='Original query'),
    'timestamp': fields.String(description='Response timestamp'),
    'pipeline_metadata': fields.Raw(description='Metadata about activated components in the processing pipeline')
})

system_status = api.model('SystemStatus', {
    'status': fields.String(description='System status'),
    'message': fields.String(description='Status message'),
    'components': fields.Raw(description='Component status details')
})

# Initialize components
config = AppConfig()
config.use_chroma_db = True  # Ensure we're using ChromaDB

# Set up ChromaParser with robust error handling
try:
    config.chroma_parser = ChromaParser(base_dir="vector_db_separate")
    # Verify that at least one collection was loaded
    collections = list(config.chroma_parser.collection_data.keys())
    if collections:
        logger.info(f"Successfully loaded ChromaParser with collections: {', '.join(collections)}")
    else:
        logger.warning("No collections found in ChromaParser, will fall back to CSV data")
except Exception as e:
    logger.error(f"Error initializing ChromaParser: {str(e)}")
    # Create a minimal ChromaParser that won't cause errors
    config.chroma_parser = ChromaParser(base_dir="vector_db_separate")
    logger.warning("Using minimal ChromaParser due to initialization error")

query_enhancer = QueryEnhancer(config)
data_router = DataRouter(config)
rag_agent = RAGAgent(config)
analyzer = AnalyticalAgent(config)
report_generator = ReportGenerator(config)
session_manager = SessionManager()
web_search_agent = WebSearchAgent(config)

@app.route('/')
def index():
    """Render the main UI page."""
    return render_template('index.html')

@app.route('/debug')
def debug():
    """Render the debug page."""
    return render_template('debug.html')

@ns_chat.route('/')
class Chat(Resource):
    @ns_chat.expect(chat_input)
    @ns_chat.marshal_with(chat_response)
    @limiter.limit("10 per minute")
    def post(self):
        """Process a chat query and return a response."""
        try:
            # Get request data
            data = request.json
            query = data.get('query')
            session_id = data.get('session_id')
            
            if not query:
                return {
                    'status': 'error',
                    'response': 'No query provided',
                    'session_id': session_id,
                    'query': '',
                    'timestamp': datetime.now().isoformat()
                }, 400
                
            if not session_id:
                session_id = str(uuid.uuid4())
                logger.info(f"Created new session: {session_id}")
            
            # Log the query
            logger.info(f"Received query: '{query}' for session {session_id}")
            
            # Process the query with pipeline components
            start_time = datetime.now()
            
            # 1. Enhance the query
            enhancement_result = query_enhancer.enhance(query)
            if enhancement_result['status'] == 'error':
                logger.error(f"Query enhancement failed: {enhancement_result['message']}")
                enhanced_query = query  # Fall back to original query
                is_web_search = False
                entities = None
                query_intent = None
            else:
                enhanced_query = enhancement_result['enhanced_query']
                is_web_search = enhancement_result.get('is_web_search', False)
                entities = enhancement_result.get('entities')
                query_intent = enhancement_result.get('query_intent')
                logger.info(f"Enhanced query: '{enhanced_query}'")
                if is_web_search:
                    logger.info("Query identified as requiring web search")
                if entities:
                    logger.info(f"Detected entities: {entities}")
                if query_intent:
                    logger.info(f"Query intent: {query_intent}")
            
            # 2. Route the query to appropriate data sources
            routing_result = data_router.route_query(enhanced_query, enhancement_result)
            if routing_result['status'] == 'error':
                logger.error(f"Query routing failed: {routing_result['message']}")
                return {
                    'status': 'error',
                    'response': f"I encountered an error while processing your query: {routing_result['message']}",
                    'session_id': session_id,
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }, 500
            
            # If query was identified as needing web search, make sure web is in sources
            if is_web_search and 'web' not in routing_result['sources']:
                logger.info("Adding web source to routing result for web search query")
                routing_result['sources'].append('web')
                # Generate web context if not already present
                if not routing_result.get('web_context'):
                    routing_result['web_context'] = web_search_agent.generate_web_context(enhanced_query)
                
            # Get data sources and data
            data_sources = routing_result['sources']
            data_dict = routing_result['data']
            additional_context = routing_result.get('transport_knowledge')
            web_context = routing_result.get('web_context')
            query_type = routing_result.get('query_type', 'factual')  # Get query type from router
            agent_type = routing_result.get('agent_type', 'vector_rag')  # Get agent type from router
            routing_confidence = routing_result.get('confidence', 0.5)  # Get confidence from router
            used_chroma_db = routing_result.get('used_chroma_db', False)  # Check if ChromaDB was used
            
            logger.info(f"Query routed to sources: {data_sources}")
            logger.info(f"Query type: {query_type}, Agent type: {agent_type}, Confidence: {routing_confidence}")
            
            # 3. Choose appropriate agent based on query type and agent type
            response_result = None
            
            # For analytical queries, use the analytical agent
            if agent_type == 'analytics':
                logger.info("Using analytical agent for response generation")
                try:
                    response_result = analyzer.analyze(
                        data=data_dict,
                        query=enhanced_query
                    )
                    # Ensure response_result has the expected structure
                    if not response_result or 'response' not in response_result:
                        logger.error("Analytics agent returned invalid response format")
                        response_result = {
                            'status': 'success',
                            'response': "I analyzed the data but couldn't generate a comprehensive analysis. Here's what I found: " + 
                                        str(data_dict.keys()) + " were examined but no specific insights could be generated.",
                            'analysis': {},
                            'charts': []
                        }
                except Exception as e:
                    logger.error(f"Error in analytics processing: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Provide a fallback response
                    response_result = {
                        'status': 'success',
                        'response': f"I tried to analyze your query but encountered a technical issue. I found data sources: {', '.join(data_sources)} but couldn't complete the analysis. You might want to try rephrasing your query or providing more specific details.",
                        'analysis': {},
                        'charts': []
                    }
            # For hybrid queries, use both RAG and analytical capabilities
            elif agent_type == 'hybrid':
                logger.info("Using hybrid approach with both RAG and analytical agents")
                # First get RAG response for guidelines/regulations
                rag_response = rag_agent.generate_response(
                    query=enhanced_query,
                    data_dict=data_dict,
                    additional_context=additional_context,
                    web_context=web_context,
                    query_type=query_type,
                    agent_type=agent_type,
                    entities=entities
                )
                
                # Then add analytical component for data
                analytical_context = rag_response.get('response', '')
                try:
                    response_result = analyzer.analyze(
                        data=data_dict,
                        query=enhanced_query
                    )
                    # If we have both RAG and analysis responses, combine them
                    if 'response' in rag_response and 'response' in response_result:
                        response_result['response'] = "Based on guidelines and regulations:\n\n" + \
                            rag_response['response'] + "\n\n" + \
                            "Data analysis shows:\n\n" + \
                            response_result['response']
                except Exception as e:
                    logger.error(f"Error in hybrid analytics processing: {str(e)}")
                    # Use RAG response if analytics fails
                    response_result = rag_response
            # For factual and other queries, use the RAG agent
            else:
                logger.info("Using RAG agent for response generation")
                response_result = rag_agent.generate_response(
                    query=enhanced_query,
                    data_dict=data_dict,
                    additional_context=additional_context,
                    web_context=web_context,
                    query_type=query_type,
                    agent_type=agent_type,
                    entities=entities
                )
            
            # Check for generation error
            if response_result.get('status') == 'error':
                logger.error(f"Response generation failed: {response_result.get('message')}")
                return {
                    'status': 'error',
                    'response': response_result.get('response', "I was unable to generate a good response to your query."),
                    'session_id': session_id,
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }, 500
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            
            # Record successful interaction
            session_manager.add_interaction(
                session_id=session_id,
                query=query,
                enhanced_query=enhanced_query,
                response=response_result['response'],
                sources=data_sources,
                query_type=query_type
            )
            
            # Return the response
            return {
                'status': 'success',
                'response': response_result['response'],
                'session_id': session_id,
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'charts': response_result.get('charts', []),
                'source_info': response_result.get('source_info', ''),
                'pipeline_metadata': {
                    'active_agent': agent_type,
                    'active_sources': data_sources,
                    'query_type': query_type,
                    'routing_confidence': routing_confidence,
                    'entities_detected': entities if entities else {}
                }
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in chat processing: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'response': "I encountered an unexpected error while processing your request.",
                'session_id': session_id if 'session_id' in locals() else str(uuid.uuid4()),
                'query': query if 'query' in locals() else "",
                'timestamp': datetime.now().isoformat()
            }, 500

@ns_system.route('/status')
class SystemStatus(Resource):
    @ns_system.marshal_with(system_status)
    def get(self):
        """Get the current system status."""
        try:
            # Check component statuses
            component_statuses = {}
            
            # Check if OpenAI API is accessible
            try:
                client = OpenAI(api_key=config.OPENAI_API_KEY)
                client.models.list()
                component_statuses['openai_api'] = {
                    'status': 'online',
                    'message': 'OpenAI API is accessible'
                }
            except Exception as e:
                component_statuses['openai_api'] = {
                    'status': 'error',
                    'message': f'OpenAI API error: {str(e)}'
                }
            
            # Check if vector collections are accessible
            try:
                # Initialize ChromaParser
                chroma_parser = config.chroma_parser or ChromaParser(base_dir="vector_db_separate")
                
                # Get collection data
                collections = list(chroma_parser.collection_data.keys())
                
                if collections:
                    collection_statuses = {}
                    for collection_name in collections:
                        try:
                            # Get document count
                            docs = chroma_parser.get_collection_documents(collection_name)
                            collection_statuses[collection_name] = {
                                'status': 'available',
                                'count': len(docs),
                                'path': os.path.join("vector_db_separate", collection_name)
                            }
                        except Exception as e:
                            collection_statuses[collection_name] = {
                                'status': 'error',
                                'message': str(e)
                            }
                    
                    component_statuses['vector_collections'] = {
                        'status': 'online',
                        'message': f'Found {len(collections)} collections',
                        'collections': collection_statuses
                    }
                else:
                    component_statuses['vector_collections'] = {
                        'status': 'warning',
                        'message': 'No vector collections found'
                    }
            except Exception as e:
                component_statuses['vector_collections'] = {
                    'status': 'error',
                    'message': f'Error accessing vector collections: {str(e)}'
                }
            
            # Overall system status
            if any(s.get('status') == 'error' for s in component_statuses.values() if isinstance(s, dict)):
                system_status = 'warning'
                message = 'Some components have issues'
            else:
                system_status = 'online'
                message = 'All systems operational'
            
            return {
                'status': system_status,
                'message': message,
                'components': component_statuses
            }
            
        except Exception as e:
            logger.error(f"Error checking system status: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Error checking system status: {str(e)}',
                'components': {}
            }, 500

@ns_chat.route('/history/<string:session_id>')
class ChatHistory(Resource):
    def get(self, session_id):
        """Get chat history for a session."""
        try:
            history = session_manager.get_history(session_id)
            return {
                'status': 'success',
                'session_id': session_id,
                'history': history
            }
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Error retrieving chat history: {str(e)}',
                'session_id': session_id,
                'history': []
            }, 500

@ns_chat.route('/clear/<string:session_id>')
class ClearChatHistory(Resource):
    def delete(self, session_id):
        """Clear chat history for a session."""
        try:
            session_manager.clear_history(session_id)
            return {
                'status': 'success',
                'message': f'History cleared for session {session_id}',
                'session_id': session_id
            }
        except Exception as e:
            logger.error(f"Error clearing chat history: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Error clearing chat history: {str(e)}',
                'session_id': session_id
            }, 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting server on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug) 