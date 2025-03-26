from flask import Flask, request, jsonify, render_template, redirect
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
import pandas as pd
import numpy as np
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Import our components
from agents.query_enhancer import QueryEnhancer
from agents.router import Router
from agents.rag import RAGAgent
from agents.analyzer import AnalyticalAgent
from agents.report_generator import ReportGenerator
from utils.session_manager import SessionManager
from config.settings import AppConfig

# Load environment variables
load_dotenv()

def check_required_env_vars():
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def check_required_files():
    required_files = [
        'embedded_data/inventory_data_embedded.csv',
        'embedded_data/transport_history_embedded.csv',
        'embedded_data/us_government_guidelines_for_medicine_transportation_and_storage_embedded.csv',
        'embedded_data/inventory_management_policy_embedded.csv'
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing required data files: {', '.join(missing_files)}")

# Perform initial checks
try:
    check_required_env_vars()
    check_required_files()
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
api = Api(app, version='1.0', title='Inventory RAG API',
    description='A RAG-based API for inventory management and analysis',
    doc='/api/docs',
    prefix='/api'  # Add prefix for all API endpoints
)

# Define namespaces
ns_chat = api.namespace('chat', description='Chat operations')
ns_system = api.namespace('system', description='System operations')
ns_inventory = api.namespace('inventory', description='Inventory operations')

# Define models for request/response validation
chat_input = api.model('ChatInput', {
    'query': fields.String(required=True, description='User query'),
    'session_id': fields.String(required=True, description='Session identifier')
})

chat_response = api.model('ChatResponse', {
    'status': fields.String(description='Response status'),
    'response': fields.Raw(description='Generated response'),
    'message': fields.String(description='Error message if any')
})

system_status = api.model('SystemStatus', {
    'status': fields.String(description='System status'),
    'message': fields.String(description='Status message'),
    'components': fields.Raw(description='Component status details')
})

inventory_stats = api.model('InventoryStats', {
    'total_items': fields.Integer(description='Total number of items'),
    'low_stock_items': fields.Integer(description='Number of items below reorder point'),
    'out_of_stock_items': fields.Integer(description='Number of items with zero stock'),
    'total_value': fields.Float(description='Total inventory value')
})

# Initialize components
try:
    config = AppConfig()
    if not config.OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set")
        
    query_enhancer = QueryEnhancer()
    router = Router()
    rag_agent = RAGAgent()
    analytical_agent = AnalyticalAgent()
    report_generator = ReportGenerator()
    session_manager = SessionManager()
    
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    raise

@app.route('/')
def index():
    """Render the main application interface"""
    return render_template('index.html')

@ns_chat.route('/')
class ChatEndpoint(Resource):
    @ns_chat.expect(chat_input)
    @ns_chat.marshal_with(chat_response)
    @limiter.limit("10/minute")
    def post(self):
        try:
            data = request.json
            if not data:
                logger.error("No data provided in request")
                return {
                    'status': 'error',
                    'message': 'No data provided in request'
                }, 400
                
            user_query = data.get('query')
            session_id = data.get('session_id')
            
            if not user_query:
                logger.error("No query provided")
                return {
                    'status': 'error',
                    'message': 'No query provided'
                }, 400
                
            if not session_id:
                logger.error("No session ID provided")
                return {
                    'status': 'error',
                    'message': 'No session ID provided'
                }, 400
            
            # Process the query through our pipeline
            logger.info(f"Processing query: {user_query} for session: {session_id}")
            
            try:
                logger.debug("Enhancing query...")
                enhanced_query = query_enhancer.enhance(user_query)
                logger.info(f"Enhanced query: {enhanced_query}")
            except Exception as e:
                logger.error(f"Error during query enhancement: {str(e)}")
                logger.error(traceback.format_exc())
                return {
                    'status': 'error',
                    'message': f'Error enhancing query: {str(e)}'
                }, 500
            
            try:
                logger.debug("Routing query...")
                selected_source = router.route(enhanced_query)
                logger.info(f"Selected source: {selected_source}")
            except Exception as e:
                logger.error(f"Error during query routing: {str(e)}")
                return {
                    'status': 'error',
                    'message': 'Error routing query. Please try again.'
                }, 500
            
            try:
                logger.debug("Processing with RAG...")
                rag_response = rag_agent.process(enhanced_query, selected_source)
                logger.info(f"RAG response generated")
            except Exception as e:
                logger.error(f"Error during RAG processing: {str(e)}")
                return {
                    'status': 'error',
                    'message': 'Error retrieving information. Please try again.'
                }, 500
            
            try:
                logger.debug("Analyzing response...")
                analysis_result = analytical_agent.analyze(rag_response)
                logger.info(f"Analysis completed")
            except Exception as e:
                logger.error(f"Error during analysis: {str(e)}")
                analysis_result = {
                    'insights': rag_response,
                    'charts': [],
                    'has_numerical_data': False
                }
            
            try:
                logger.debug("Generating final report...")
                final_report = report_generator.generate(analysis_result)
                logger.info(f"Report generated")
            except Exception as e:
                logger.error(f"Error generating report: {str(e)}")
                return {
                    'status': 'error',
                    'message': 'Error generating report. Please try again.'
                }, 500
            
            try:
                # Store in session history
                session_manager.add_to_history(session_id, {
                    'query': user_query,
                    'response': final_report
                })
            except Exception as e:
                logger.error(f"Error storing in session history: {str(e)}")
            
            return {
                'status': 'success',
                'response': final_report
            }
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': 'An unexpected error occurred. Please try again.',
                'details': str(e) if app.debug else None
            }, 500

@ns_system.route('/status')
class SystemStatusEndpoint(Resource):
    @ns_system.marshal_with(system_status)
    def get(self):
        try:
            components = {
                'query_enhancer': query_enhancer is not None,
                'router': router is not None,
                'rag_agent': rag_agent is not None,
                'analytical_agent': analytical_agent is not None,
                'report_generator': report_generator is not None,
                'session_manager': session_manager is not None
            }
            
            all_healthy = all(components.values())
            
            return {
                'status': 'success' if all_healthy else 'degraded',
                'message': 'System online' if all_healthy else 'Some components are unavailable',
                'components': components
            }
        except Exception as e:
            logger.error(f"Error checking system status: {str(e)}")
            return {
                'status': 'error',
                'message': 'Error checking system status'
            }, 500

@ns_inventory.route('/stats')
class InventoryStats(Resource):
    @ns_inventory.marshal_with(inventory_stats)
    @cache.cached(timeout=60)  # Cache for 1 minute
    def get(self):
        try:
            df = pd.read_csv('embedded_data/inventory_data_embedded.csv')
            if df is None:
                return {'message': 'Error loading inventory data'}, 500
                
            stats = {
                'total_items': len(df),
                'low_stock_items': len(df[df['CurrentStock'] <= df['ReorderPoint']]),
                'out_of_stock_items': len(df[df['CurrentStock'] == 0]),
                'total_value': (df['CurrentStock'] * df['UnitCost']).sum()
            }
            return stats
        except Exception as e:
            logger.error(f"Error calculating inventory stats: {str(e)}")
            return {'message': 'Error calculating inventory stats'}, 500

@ns_inventory.route('/low-stock')
class LowStockItems(Resource):
    @cache.cached(timeout=60)  # Cache for 1 minute
    def get(self):
        try:
            df = pd.read_csv('embedded_data/inventory_data_embedded.csv')
            if df is None:
                return {'message': 'Error loading inventory data'}, 500
                
            low_stock = df[df['CurrentStock'] <= df['ReorderPoint']]
            items = low_stock[[
                'ItemID', 'GenericName', 'CurrentStock', 
                'ReorderPoint', 'Unit', 'Status'
            ]].to_dict('records')
            
            return {
                'status': 'success',
                'count': len(items),
                'items': items
            }
        except Exception as e:
            logger.error(f"Error retrieving low stock items: {str(e)}")
            return {'message': 'Error retrieving low stock items'}, 500

@ns_inventory.route('/search')
class InventorySearch(Resource):
    @ns_inventory.param('query', 'Search query string')
    def get(self):
        try:
            query = request.args.get('query', '').lower()
            if not query:
                return {'message': 'No search query provided'}, 400
                
            df = pd.read_csv('embedded_data/inventory_data_embedded.csv')
            if df is None:
                return {'message': 'Error loading inventory data'}, 500
                
            # Search in GenericName and Status
            mask = df['GenericName'].str.lower().str.contains(query) | \
                   df['Status'].str.lower().str.contains(query)
            results = df[mask][[
                'ItemID', 'GenericName', 'CurrentStock', 
                'Unit', 'Status', 'UnitCost', 'SellingPrice'
            ]].to_dict('records')
            
            return {
                'status': 'success',
                'count': len(results),
                'items': results
            }
        except Exception as e:
            logger.error(f"Error searching inventory: {str(e)}")
            return {'message': 'Error searching inventory'}, 500

@ns_inventory.route('/comprehensive')
class ComprehensiveInventory(Resource):
    @cache.cached(timeout=60)  # Cache for 1 minute
    def get(self):
        """Get comprehensive inventory data with classification-based prioritization"""
        try:
            # Load inventory data
            df = pd.read_csv('embedded_data/inventory_data_embedded.csv')
            if df is None:
                return {'message': 'Error loading inventory data'}, 500
            
            # Load policy data for classification rules
            policy_df = pd.read_csv('embedded_data/inventory_management_policy_embedded.csv')
            
            # Classify items based on policy rules
            def classify_item(row):
                if row['UnitCost'] * row['CurrentStock'] > 100000:  # High value
                    return 'A'
                elif row['UnitCost'] * row['CurrentStock'] > 50000:  # Moderate value
                    return 'B'
                return 'C'  # Low value
            
            df['Classification'] = df.apply(classify_item, axis=1)
            
            # Add storage condition flags
            df['RequiresRefrigeration'] = df['StorageCondition'].str.contains('Refrigerated', case=False)
            df['RequiresSpecialHandling'] = df['SpecialHandling'].notna()
            
            # Calculate stock status
            df['StockStatus'] = 'Normal'
            df.loc[df['CurrentStock'] <= df['ReorderPoint'], 'StockStatus'] = 'Low'
            df.loc[df['CurrentStock'] == 0, 'StockStatus'] = 'Out of Stock'
            
            # Prepare comprehensive item details
            items = []
            
            # Process A-class items first (critical items)
            a_items = df[df['Classification'] == 'A'].sort_values('CurrentStock')
            for _, item in a_items.iterrows():
                items.append({
                    'ItemID': item['ItemID'],
                    'GenericName': item['GenericName'],
                    'Classification': 'A',
                    'CurrentStock': item['CurrentStock'],
                    'MaxInventory': item['MaxInventory'],
                    'ReorderPoint': item['ReorderPoint'],
                    'Unit': item['Unit'],
                    'StorageCondition': item['StorageCondition'],
                    'SpecialHandling': item['SpecialHandling'],
                    'UnitCost': item['UnitCost'],
                    'SellingPrice': item['SellingPrice'],
                    'LeadTimeDays': item['LeadTimeDays'],
                    'Status': item['Status'],
                    'StockStatus': item['StockStatus'],
                    'LastUpdated': item['LastUpdated']
                })
            
            # Process B-class items next
            b_items = df[df['Classification'] == 'B'].sort_values('CurrentStock')
            for _, item in b_items.iterrows():
                items.append({
                    'ItemID': item['ItemID'],
                    'GenericName': item['GenericName'],
                    'Classification': 'B',
                    'CurrentStock': item['CurrentStock'],
                    'MaxInventory': item['MaxInventory'],
                    'ReorderPoint': item['ReorderPoint'],
                    'Unit': item['Unit'],
                    'StorageCondition': item['StorageCondition'],
                    'SpecialHandling': item['SpecialHandling'],
                    'UnitCost': item['UnitCost'],
                    'SellingPrice': item['SellingPrice'],
                    'LeadTimeDays': item['LeadTimeDays'],
                    'Status': item['Status'],
                    'StockStatus': item['StockStatus'],
                    'LastUpdated': item['LastUpdated']
                })
            
            # Process C-class items last
            c_items = df[df['Classification'] == 'C'].sort_values('CurrentStock')
            for _, item in c_items.iterrows():
                items.append({
                    'ItemID': item['ItemID'],
                    'GenericName': item['GenericName'],
                    'Classification': 'C',
                    'CurrentStock': item['CurrentStock'],
                    'MaxInventory': item['MaxInventory'],
                    'ReorderPoint': item['ReorderPoint'],
                    'Unit': item['Unit'],
                    'StorageCondition': item['StorageCondition'],
                    'SpecialHandling': item['SpecialHandling'],
                    'UnitCost': item['UnitCost'],
                    'SellingPrice': item['SellingPrice'],
                    'LeadTimeDays': item['LeadTimeDays'],
                    'Status': item['Status'],
                    'StockStatus': item['StockStatus'],
                    'LastUpdated': item['LastUpdated']
                })
            
            # Prepare summary statistics
            summary = {
                'total_items': len(df),
                'items_by_class': {
                    'A': len(a_items),
                    'B': len(b_items),
                    'C': len(c_items)
                },
                'items_by_status': {
                    'normal': len(df[df['StockStatus'] == 'Normal']),
                    'low_stock': len(df[df['StockStatus'] == 'Low']),
                    'out_of_stock': len(df[df['StockStatus'] == 'Out of Stock'])
                },
                'special_handling': {
                    'refrigerated': df['RequiresRefrigeration'].sum(),
                    'special_handling': df['RequiresSpecialHandling'].sum()
                }
            }
            
            return {
                'status': 'success',
                'summary': summary,
                'items': items
            }
            
        except Exception as e:
            logger.error(f"Error retrieving comprehensive inventory data: {str(e)}")
            return {'message': 'Error retrieving comprehensive inventory data'}, 500

if __name__ == '__main__':
    try:
        # Test database connection
        df = pd.read_csv('embedded_data/inventory_data_embedded.csv')
        logger.info(f"Successfully loaded inventory data with {len(df)} records")
        
        # Start the application
        port = int(os.getenv('PORT', 5000))
        app.run(debug=True, port=port)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1) 