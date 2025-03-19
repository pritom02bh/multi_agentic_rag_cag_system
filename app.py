from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from dotenv import load_dotenv
import time
import logging
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import json
from agents.base_agent import BaseAgent, AgentType
from agents.agent_registry import AgentRegistry, get_agent_registry
from threading import Lock
import openai
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_session import Session
from concurrent.futures import ThreadPoolExecutor
import asyncio
from limits.storage import RedisStorage, MemoryStorage
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Initialize Flask app
app = Flask(__name__, 
           template_folder='ui_service/templates',
           static_folder='ui_service/static')
CORS(app)

# Configure session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize Redis connection
redis_url = os.getenv('REDIS_URL')
try:
    redis_client = redis.from_url(redis_url)
    redis_client.ping()
    logger.info("Successfully connected to Redis")
    storage = RedisStorage(redis_url)
except (redis.ConnectionError, redis.ResponseError) as e:
    logger.warning(f"Redis connection failed ({str(e)}), falling back to in-memory storage")
    storage = MemoryStorage()

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=redis_url if isinstance(storage, RedisStorage) else None,
    default_limits=["100 per minute"]
)

# Thread pool for async operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Global agent registry and lock
agent_registry = None
agent_lock = Lock()

# Cache for system status
status_cache = {
    'last_check': None,
    'cache_duration': 60,  # Cache for 60 seconds
    'status': None
}

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/api/ui/chat', methods=['POST'])
@limiter.limit("20 per minute")
async def chat():
    """Handle chat requests and route to appropriate agents using the updated workflow."""
    try:
        data = request.get_json()
        if not data:
            logger.error("No data provided in request")
            return jsonify({
                'error': 'No data provided',
                'type': 'error'
            }), 400
            
        query = data.get('query', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        logger.info(f"Processing chat request - query: '{query}', session_id: {session_id}")
        
        if not query:
            logger.error("Empty query in request")
            return jsonify({
                'error': 'Query is required',
                'type': 'error'
            }), 400
        
        # Get agent registry
        try:
            registry = get_agent_registry()
            logger.info("Agent registry retrieved successfully")
        except Exception as e:
            logger.error(f"Failed to get agent registry: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Failed to initialize agent system',
                'type': 'error'
            }), 500
        
        # Initialize pipeline status
        pipeline_status = {
            'query_received': 'completed',
            'query_enhancement': 'waiting',
            'router_agent': 'waiting',
            'knowledge_base': 'waiting',
            'policy_transport': 'waiting',
            'response_generation': 'waiting',
            'ui_response': 'waiting'
        }
        
        try:
            # Step 1: Query Enhancement
            pipeline_status['query_enhancement'] = 'in_progress'
            router_agent = registry.get_agent(AgentType.ROUTER)
            if not router_agent:
                logger.error("Router agent not available")
                return jsonify({
                    'error': 'Router agent not available',
                    'type': 'error',
                    'pipeline_status': pipeline_status
                }), 500
                
            pipeline_status['query_enhancement'] = 'completed'
            
            # Step 2: Route and Process Query
            pipeline_status['router_agent'] = 'in_progress'
            
            # Process with router agent that handles the whole workflow
            logger.info(f"Processing query with router agent: {query}")
            try:
                response = await router_agent.process_query(query)
                logger.info(f"Router agent response received. Response type: {type(response)}")
                if isinstance(response, dict):
                    logger.info(f"Router agent response keys: {response.keys()}")
                    if 'error' in response and response['error']:
                        logger.error(f"Error in router agent response: {response['error']}")
                else:
                    logger.warning(f"Non-dict response: {response}")
            except Exception as e:
                logger.error(f"Error calling router_agent.process_query: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'error': f'Error processing query: {str(e)}',
                    'type': 'error',
                    'pipeline_status': pipeline_status
                }), 500
            
            # Update pipeline status based on response metadata
            if isinstance(response, dict):
                if 'metadata' in response and 'query_type' in response['metadata']:
                    query_type = response['metadata']['query_type']
                    logger.info(f"Query type identified: {query_type}")
                    
                    if query_type == 'inventory':
                        pipeline_status['knowledge_base'] = 'completed'
                        pipeline_status['policy_transport'] = 'skipped'
                    elif query_type in ['transport', 'policy', 'guidelines']:
                        pipeline_status['knowledge_base'] = 'skipped'
                        pipeline_status['policy_transport'] = 'completed'
                    else:
                        pipeline_status['knowledge_base'] = 'completed'
                        pipeline_status['policy_transport'] = 'completed'
            
            pipeline_status['router_agent'] = 'completed'
            pipeline_status['response_generation'] = 'completed'
            
            # Check if we received a valid response
            if not response:
                logger.error("Empty response from router agent")
                return jsonify({
                    'error': 'Empty response from router agent',
                    'type': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_status': pipeline_status
                }), 500
                
            if not isinstance(response, dict):
                logger.error(f"Invalid response type from router agent: {type(response)}")
                # Try to convert non-dict response to dict
                try:
                    if isinstance(response, str):
                        response = {
                            'response': response,
                            'type': 'assistant'
                        }
                        logger.info("Converted string response to dict")
                    else:
                        logger.error(f"Cannot convert response of type {type(response)} to dict")
                        return jsonify({
                            'error': f'Invalid response type from router agent: {type(response)}',
                            'type': 'error',
                            'timestamp': datetime.now().isoformat(),
                            'pipeline_status': pipeline_status
                        }), 500
                except Exception as e:
                    logger.error(f"Error converting response: {str(e)}")
                    return jsonify({
                        'error': f'Error converting response: {str(e)}',
                        'type': 'error',
                        'timestamp': datetime.now().isoformat(),
                        'pipeline_status': pipeline_status
                    }), 500
            
            # Prepare UI response
            pipeline_status['ui_response'] = 'in_progress'
            
            # Determine the main content
            main_content = ""
            if 'response' in response:
                main_content = response['response']
            elif 'summary' in response:
                main_content = response['summary']
            elif 'content' in response:
                main_content = response['content']
            
            # Check for error
            error_occurred = False
            if 'error' in response and response['error']:
                error_occurred = True
                if not main_content:
                    main_content = f"Error: {response['error']}"
                logger.error(f"Error in response: {response['error']}")
            
            if not main_content:
                logger.warning("No main content found in response, using empty string")
            
            logger.info(f"Main content length: {len(main_content) if main_content else 0}")
            
            # Create a detailed response that includes all available information
            ui_response = {
                'content': main_content,
                'type': 'assistant',
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'pipeline_status': pipeline_status
            }
            
            # If this is an error response, add appropriate flags
            if error_occurred:
                ui_response['error'] = True
                ui_response['error_type'] = 'system_error' if 'rate_limit' in str(response.get('error', '')) else 'processing_error'
            
            # Add context when available
            if 'context' in response:
                logger.info("Adding context information to UI response")
                ui_response['context'] = response['context']
            
            # Add raw data when available
            if 'data' in response:
                logger.info("Adding raw data to UI response")
                ui_response['data'] = response['data']
            
            # Add visualization if available
            if 'visualization' in response:
                ui_response['visualization'] = response['visualization']
                
            # Add table data if available
            if 'table_data' in response:
                ui_response['table_data'] = response['table_data']
                
            # Add details if available
            if 'details' in response and response['details']:
                ui_response['details'] = response['details']
            
            # Add analysis if available
            if 'analysis' in response and response['analysis']:
                ui_response['analysis'] = response['analysis']
            
            # Add recommendations if available
            if 'recommendations' in response and response['recommendations']:
                ui_response['recommendations'] = response['recommendations']
                
            # Add sources if available
            if 'sources' in response:
                ui_response['sources'] = response['sources']
            elif 'metadata' in response and 'sources' in response['metadata']:
                ui_response['sources'] = response['metadata']['sources']
            
            # Don't try to enhance error responses
            should_enhance = not error_occurred
                
            # For inventory-specific queries, include the detailed information in formatted form
            if should_enhance and isinstance(response, dict) and ('inventory' in query.lower() or 
                    (response.get('metadata', {}).get('query_type', '') == 'inventory')):
                logger.info("Enhancing inventory response with structured data")
                
                try:
                    # Extract more detailed information from the context if available
                    if 'context' in response:
                        # Format the content to be more user-friendly and detailed
                        context_data = response['context']
                        # Keep the original content, but add a more comprehensive detailed section
                        enhanced_details = _format_inventory_details(context_data, main_content)
                        if enhanced_details:
                            ui_response['content'] = enhanced_details
                except Exception as format_error:
                    logger.error(f"Error formatting inventory details: {str(format_error)}")
                    logger.error(traceback.format_exc())
                
            pipeline_status['ui_response'] = 'completed'
            logger.info(f"Final UI response keys: {ui_response.keys()}")
            
            return jsonify(ui_response)
            
        except Exception as e:
            logger.error(f"Error in chat pipeline: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'error': f'An error occurred while processing your request: {str(e)}',
                'type': 'error',
                'timestamp': datetime.now().isoformat(),
                'pipeline_status': pipeline_status
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}',
            'type': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get system status including agent availability with caching."""
    current_time = time.time()
    
    # Return cached status if available and not expired
    if (status_cache['last_check'] is not None and 
        current_time - status_cache['last_check'] < status_cache['cache_duration'] and
        status_cache['status'] is not None):
        return jsonify(status_cache['status'])
    
    try:
        registry = get_agent_registry()
        available_agents = [
            agent_type.value for agent_type in AgentType 
            if registry.get_agent(agent_type) is not None
        ]
        
        # Update cache
        status_data = {
            'status': 'online',
            'available_agents': available_agents,
            'timestamp': datetime.now().isoformat()
        }
        status_cache['status'] = status_data
        status_cache['last_check'] = current_time
        
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('ui_service/static', path)

def _format_inventory_details(context_data, original_content):
    """Format inventory details in a more comprehensive and readable way."""
    try:
        if not context_data:
            return original_content
            
        # If the content is already detailed enough, return it
        if len(original_content.split('\n')) > 10:
            return original_content
            
        # Try to parse context data to extract more details
        import re
        
        # Extract inventory items from context
        items = []
        item_data = {}
        current_item = {}
        
        # Look for JSON-like data in the context
        # This pattern matches both {'key': 'value'} structures and more general content
        json_pattern = r"'([^']+)': '([^']+)'"
        item_id_pattern = r"'ItemID': '([^']+)'"
        
        # First identify item blocks and process them properly
        for line in context_data.split('\n'):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Start of a new item block
            if line.startswith('From') or line.startswith('\nFrom'):
                if current_item and 'ItemID' in current_item:
                    items.append(current_item)
                current_item = {}
                continue
                
            # Extract JSON data from the line
            if '{' in line and '}' in line:
                # Extract all key-value pairs
                matches = re.findall(json_pattern, line)
                if matches:
                    for key, value in matches:
                        current_item[key] = value
        
        # Add the last item if it exists
        if current_item and 'ItemID' in current_item:
            items.append(current_item)
            
        # If we couldn't extract structured data from JSON, try direct parsing
        if not items:
            # Try to extract structured data directly from formatted content
            current_item = {}
            lines = context_data.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Look for item ID pattern
                id_match = re.search(r'Item ID: ([A-Z0-9-]+)', line)
                if id_match:
                    if current_item and 'ItemID' in current_item:
                        items.append(current_item)
                    current_item = {'ItemID': id_match.group(1)}
                    continue
                    
                # Look for other details
                if 'Batch Number:' in line:
                    current_item['BatchNumber'] = line.split('Batch Number:')[1].strip()
                elif 'Current Stock:' in line:
                    stock_parts = line.split('Current Stock:')[1].strip().split()
                    current_item['CurrentStock'] = stock_parts[0].replace(',', '')
                    if len(stock_parts) > 1:
                        current_item['Unit'] = stock_parts[1]
                elif 'Expiry Date:' in line:
                    expiry_date = line.split('Expiry Date:')[1].strip()
                    # Try to convert to standardized format
                    try:
                        from datetime import datetime
                        months = {
                            'January': '01', 'February': '02', 'March': '03', 'April': '04',
                            'May': '05', 'June': '06', 'July': '07', 'August': '08',
                            'September': '09', 'October': '10', 'November': '11', 'December': '12'
                        }
                        parts = expiry_date.replace(',', '').split()
                        if len(parts) == 3:
                            month, day, year = parts[0], parts[1], parts[2]
                            current_item['ExpiryDate'] = f"{year}-{months.get(month, '01')}-{day.zfill(2)}"
                        else:
                            current_item['ExpiryDate'] = expiry_date
                    except:
                        current_item['ExpiryDate'] = expiry_date
                elif 'Storage Condition:' in line:
                    cond_parts = line.split('Storage Condition:')[1].strip().split('(')
                    current_item['StorageCondition'] = cond_parts[0].strip()
                    if len(cond_parts) > 1 and ')' in cond_parts[1]:
                        current_item['SpecialHandling'] = cond_parts[1].split(')')[0].strip()
        
            # Add the last item if it exists
            if current_item and 'ItemID' in current_item:
                items.append(current_item)
            
        # If we still couldn't extract data, return the original content
        if not items:
            return original_content
            
        # Format the items into a detailed list
        formatted_content = "Based on the information from our medical supply chain database, here is a detailed list of pharmaceutical items in our inventory:\n\n"
        
        for i, item in enumerate(items, 1):
            formatted_content += f"{i}. Item ID: {item.get('ItemID', 'Unknown')}\n"
            
            if 'BatchNumber' in item:
                formatted_content += f"   - Batch Number: {item['BatchNumber']}\n"
                
            if 'CurrentStock' in item:
                # Add units if available
                unit = item.get('Unit', '')
                formatted_content += f"   - Current Stock: {item['CurrentStock']} {unit}\n"
                
            if 'ExpiryDate' in item:
                # Convert date format if needed (YYYY-MM-DD to more readable format)
                try:
                    from datetime import datetime
                    if '-' in item['ExpiryDate']:
                        date_obj = datetime.strptime(item['ExpiryDate'], '%Y-%m-%d')
                        formatted_date = date_obj.strftime('%B %d, %Y')
                        formatted_content += f"   - Expiry Date: {formatted_date}\n"
                    else:
                        formatted_content += f"   - Expiry Date: {item['ExpiryDate']}\n"
                except:
                    formatted_content += f"   - Expiry Date: {item['ExpiryDate']}\n"
                    
            if 'StorageCondition' in item:
                special = item.get('SpecialHandling', '')
                if special:
                    formatted_content += f"   - Storage Condition: {item['StorageCondition']} ({special})\n"
                else:
                    formatted_content += f"   - Storage Condition: {item['StorageCondition']}\n"
                    
            # Add additional details that might be useful
            if 'ReorderPoint' in item:
                # Format as a number with commas for thousands
                try:
                    reorder_point = float(item['ReorderPoint'])
                    formatted_content += f"   - Reorder Point: {reorder_point:,.0f}\n"
                except:
                    formatted_content += f"   - Reorder Point: {item['ReorderPoint']}\n"
                
            if 'LeadTimeDays' in item:
                formatted_content += f"   - Lead Time: {item['LeadTimeDays']} days\n"
                
            if 'UnitCost' in item:
                formatted_content += f"   - Unit Cost: ${item['UnitCost']}\n"
                
            if 'SellingPrice' in item:
                formatted_content += f"   - Selling Price: ${item['SellingPrice']}\n"
                
            if 'ManufacturingDate' in item:
                try:
                    from datetime import datetime
                    if '-' in item['ManufacturingDate']:
                        date_obj = datetime.strptime(item['ManufacturingDate'], '%Y-%m-%d')
                        formatted_date = date_obj.strftime('%B %d, %Y')
                        formatted_content += f"   - Manufacturing Date: {formatted_date}\n"
                    else:
                        formatted_content += f"   - Manufacturing Date: {item['ManufacturingDate']}\n"
                except:
                    formatted_content += f"   - Manufacturing Date: {item['ManufacturingDate']}\n"
                    
            if 'Status' in item:
                formatted_content += f"   - Status: {item['Status']}\n"
                    
            # Add a newline between items
            formatted_content += "\n"
            
        formatted_content += "These details provide a comprehensive overview of the pharmaceutical items in our inventory, including their quantities, expiry dates, and specific storage conditions."
        
        return formatted_content
        
    except Exception as e:
        logger.error(f"Error in _format_inventory_details: {str(e)}")
        return original_content

if __name__ == '__main__':
    app.run(debug=True) 