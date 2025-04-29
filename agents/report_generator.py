import json
from config.settings import AppConfig
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
import traceback

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates formatted reports from RAG responses and analysis results."""
    
    def __init__(self, config=None):
        """Initialize the report generator."""
        self.config = config or AppConfig()
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            logger.info("ReportGenerator initialized with OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
        
        # Set up the system prompt for report generation - with dynamic structure
        self.system_prompt = """You are an expert report generator for pharmaceutical inventory and supply chain management.
Your task is to create clear, concise, and informative responses based on the provided information and analysis data.

IMPORTANT: You must adapt the structure and format of your response based on the query type, content, and complexity:

1. For simple factual queries (e.g., "What's the stock level of paracetamol?", "What is today's temperature?"):
   - Provide a direct, concise answer
   - No report-style formatting or sections
   - Keep formatting minimal and focus on the facts

2. For analytical queries or complex topics:
   - Create a logical structure that fits the specific content of the response
   - Organize information in a way that makes the most sense for the particular query
   - Use appropriate headings that reflect the actual content
   - Include only sections that are relevant to the specific query and findings

3. For external information from web search:
   - Present information clearly with appropriate attribution
   - Structure should be determined by the nature of the information
   - Adapt formatting to the content rather than forcing a template

FLEXIBLE FORMATTING GUIDELINES:
- Use markdown formatting to enhance readability without being rigid
- Create headings that reflect the actual content rather than generic templates
- Highlight important data using appropriate formatting techniques
- Format quantitative information consistently but adapt to the data's nature
- Organize content with logical grouping and visual separation
- For tables or lists, use the format that best presents the specific information

GENERAL PRINCIPLES:
- Focus on clarity and accuracy first
- Let the content determine the structure, not the other way around
- Adapt your formatting to enhance understanding of the specific information
- Keep responses proportionate to the complexity of the query
- Ensure information is presented in a logical flow

Never add unnecessary boilerplate text like "Here is the information you requested" or "I hope this helps."
Avoid using rigid templates that don't fit the specific content of the response."""

        self.report_template = """
Original Query: {original_query}
RAG Response: {rag_response}
Analysis Results: {analysis_results}

Generate an appropriately formatted response based on this information.

GUIDELINES FOR FLEXIBLE RESPONSE FORMATTING:
- Let the content determine the appropriate structure and organization
- Adapt your formatting style to match the query's complexity and subject matter
- For simple queries, provide direct answers without unnecessary structure
- For complex analytical content, create a logical organization that fits the specific information
- Use headings, formatting, and organization that make sense for the particular content
- Avoid forcing the content into a rigid template structure

Focus on making the information clear, well-organized, and easily understood.
Format the response in a way that best serves the specific content and query.
"""
        
        logger.info("ReportGenerator initialization complete")
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query to determine appropriate formatting.
        
        Args:
            query: The original user query
            
        Returns:
            str: The query type ('simple', 'analytical', or 'web_based')
        """
        query = query.lower()
        
        # Check for analytical/complex query indicators
        analytical_indicators = [
            'analyze', 'analysis', 'report', 'summarize', 'compare', 'comparison',
            'trends', 'performance', 'statistics', 'metrics', 'kpi', 'evaluate',
            'assessment', 'forecast', 'projection', 'predict', 'optimization',
            'strategy', 'recommendation', 'risk', 'compliance', 'audit'
        ]
        
        # Check for web search indicators
        web_search_indicators = [
            'weather', 'temperature', 'news', 'recent', 'latest', 'update',
            'current', 'today', 'regulation', 'guideline', 'policy', 'law',
            'external', 'outside', 'market', 'industry', 'global', 'worldwide'
        ]
        
        # Check if query matches analytical patterns
        for indicator in analytical_indicators:
            if indicator in query:
                logger.info(f"Query '{query}' classified as analytical based on indicator '{indicator}'")
                return 'analytical'
        
        # Check if query matches web search patterns
        for indicator in web_search_indicators:
            if indicator in query:
                logger.info(f"Query '{query}' may be web-based due to indicator '{indicator}'")
                return 'web_based'
        
        # Default to simple query if no specific indicators are found
        logger.info(f"Query '{query}' classified as simple (no specific indicators found)")
        return 'simple'
        
    def format_response(self, rag_response: str, analysis_results: Optional[Dict[str, Any]] = None, original_query: str = "", query_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Format the RAG response and analysis results into a well-structured report.
        
        Args:
            rag_response: The raw response from the RAG system
            analysis_results: Optional analysis results to incorporate
            original_query: The original user query to help determine appropriate formatting
            query_type: Optional query type from router ('simple_factual', 'analytical', etc.)
            
        Returns:
            Dict containing the formatted report and status information
        """
        try:
            # Determine query type - use provided type or detect if not provided
            determined_query_type = query_type or self._detect_query_type(original_query)
            logger.info(f"Generating formatted response for query type: {determined_query_type}")
            
            # Handle case with no analysis results
            analysis_json = "{}"
            if analysis_results:
                try:
                    # Convert analysis results to JSON string
                    analysis_json = json.dumps(analysis_results, default=str)
                except Exception as e:
                    logger.error(f"Error converting analysis results to JSON: {str(e)}")
                    analysis_json = "{}"
            
            # Generate formatted report using OpenAI
            try:
                # Add query type to template
                template = self.report_template + f"\nQuery type detected: {determined_query_type}"
                
                # Add contextual information rather than specific formatting rules
                if 'simple' in determined_query_type or 'factual' in determined_query_type:
                    template += "\nThis appears to be a simple factual query. Consider providing a concise, direct answer."
                elif 'analytical' in determined_query_type:
                    template += "\nThis appears to be an analytical query. Consider organizing the information logically based on the content."
                elif 'web' in determined_query_type:
                    template += "\nThis query may involve external information. Consider including attribution to sources when applicable."
                
                response = self.client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": template.format(
                            original_query=original_query,
                            rag_response=rag_response,
                            analysis_results=analysis_json
                        )}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                formatted_report = response.choices[0].message.content.strip()
                logger.info(f"Successfully generated formatted response for query type: {determined_query_type}")
                
                return {
                    'status': 'success',
                    'message': 'Response generated successfully',
                    'response': formatted_report,
                    'original_response': rag_response,
                    'query_type': determined_query_type
                }
                
            except Exception as e:
                logger.error(f"Error generating report with OpenAI: {str(e)}")
                logger.error(traceback.format_exc())
                # Fall back to raw response if report generation fails
                return {
                    'status': 'error',
                    'message': f'Error generating report: {str(e)}',
                    'response': rag_response,
                    'original_response': rag_response
                }
            
        except Exception as e:
            logger.error(f"Error in response formatting: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Error formatting response: {str(e)}',
                'response': rag_response,
                'original_response': rag_response
            }
    
    def generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """
        Generate a formatted error report when the main process fails.
        
        Args:
            error_message: The error message to include in the report
            
        Returns:
            Dict containing the error report
        """
        error_report = f"""An error occurred while processing your request.

{error_message}

You may want to try:
* Rephrasing your query
* Asking about different information
* Being more specific in your question
* Contacting support if the problem persists"""

        return {
            'status': 'error',
            'message': 'Error occurred during processing',
            'response': error_report,
            'original_response': error_message
        }

    def generate(self, query: str, rag_response: dict, analytics: dict = None) -> dict:
        """
        Generate a final report combining RAG response and analytics.
        
        Args:
            query (str): The original user query
            rag_response (dict): The response from the RAG agent
            analytics (dict, optional): Additional analytics if available
            
        Returns:
            dict: Formatted report with response, insights, and visualizations
        """
        try:
            # Extract components from RAG response
            response_text = rag_response.get('response', '')
            charts = rag_response.get('charts', [])
            
            # Add analytics if available
            if analytics and isinstance(analytics, dict):
                # Add any charts from analytics
                if 'charts' in analytics:
                    charts.extend(analytics.get('charts', []))
                
                # Add analytics data to the report
                analysis_data = {
                    'summary': analytics.get('summary', {}),
                    'stock_levels': analytics.get('stock_levels', {}),
                    'risks': analytics.get('risks', {}),
                    'trends': analytics.get('trends', {}),
                    'predictions': analytics.get('predictions', {})
                }
            else:
                analysis_data = {}
            
            # Format the report
            report = {
                'response': response_text,
                'charts': charts,
                'analysis': analysis_data,
                'metadata': {
                    'query': query,
                    'has_analytics': analytics is not None,
                    'has_visualizations': bool(charts),
                    'visualization_count': len(charts),
                    'generated_at': pd.Timestamp.now().isoformat()
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {
                'response': 'Error generating report',
                'charts': [],
                'analysis': {},
                'metadata': {
                    'error': str(e),
                    'generated_at': pd.Timestamp.now().isoformat()
                }
            } 