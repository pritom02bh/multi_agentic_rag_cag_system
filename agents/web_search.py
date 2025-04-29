import os
import logging
import json
import requests
from typing import Dict, Any, List, Optional
import traceback
from config.settings import AppConfig
from openai import OpenAI
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSearchAgent:
    """Web Search Agent that uses SERPER API to search the web."""
    
    def __init__(self, config=None):
        """Initialize the Web Search Agent with configuration."""
        self.config = config or AppConfig()
        
        # Add SERPER API key from environment variable or config
        self.api_key = os.getenv('SERPER_API_KEY', 'cfd71b88c74a06e4bebca4b1fc27cee45650cf32')
        
        # SERPER API endpoint
        self.api_url = "https://google.serper.dev/search"
        
        # Headers for SERPER API
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        # Initialize OpenAI client for query refinement
        try:
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            logger.info("OpenAI client initialized for query refinement")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
        
        # Track the last few queries and their results to avoid repeated searches
        self.query_cache = {}
        self.max_cache_size = 10
        
        logger.info("Web Search Agent initialized")
    
    def refine_search_query(self, query: str) -> str:
        """
        Refine the search query to improve search results using OpenAI.
        
        Args:
            query: The original search query
            
        Returns:
            Refined search query
        """
        if not self.client:
            logger.warning("OpenAI client not available for query refinement")
            return query
            
        try:
            # Check if this is a general query that shouldn't be transformed
            general_query_patterns = [
                r'\b(weather|temperature|forecast)\b',
                r'\b(time|current time)\b',
                r'\b(news|latest news)\b',
                r'\b(stock price|market value)\b',
                r'\b(sports|game|match)\b',
                r'\b(population|how many people)\b',
                r'\b(currency|exchange rate)\b',
                r'\b(movie|show times)\b',
                r'\b(traffic)\b',
                r'\b(restaurant|places to eat)\b',
                r'\b(tariff|tariffs|import duty|export duty|customs duty)\b',
                r'\b(tax rate|sales tax|vat|value added tax|import tax)\b',
                r'\b(shipping cost|transport fees|logistics charges)\b',
                r'\b(regulatory changes|regulation updates|new laws|legislation)\b',
                r'\b(market prices|commodity prices|raw material costs)\b',
                r'\b(international trade|trade agreements|trade policies)\b',
                r'\b(economic indicators|gdp|inflation rate|unemployment)\b',
                r'\b(global events|political situation|geopolitical)\b',
                r'\b(exchange rates|currency value|foreign exchange)\b',
                r'\b(supply chain disruptions|logistics issues|shipping delays)\b',
            ]
            
            # Check for specific topics that need precise query enhancement
            tariff_pattern = r'\b(tariff|tariffs|import duty|export duty|customs duty|section 301)\b'
            is_tariff_query = bool(re.search(tariff_pattern, query, re.IGNORECASE))
            
            pharmaceutical_pattern = r'\b(pharmaceutical|medicine|drug|medical|healthcare)\b'
            is_pharmaceutical_query = bool(re.search(pharmaceutical_pattern, query, re.IGNORECASE))
            
            # Use specialized enhancement for tariff queries about pharmaceuticals
            if is_tariff_query and is_pharmaceutical_query:
                logger.info("Detected pharmaceutical tariff query, using specialized enhancement")
                current_year = datetime.now().year
                enhanced_query = f"{query.strip()} {current_year} current rates latest updates"
                logger.info(f"Enhanced pharmaceutical tariff query: '{enhanced_query}' (from: '{query}')")
                return enhanced_query
            
            for pattern in general_query_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    logger.info(f"Detected general query pattern, enhancing with date/recency markers: '{query}'")
                    # Add current year and "latest" for better recency
                    current_year = datetime.now().year
                    return f"{query.strip()} {current_year} latest"
            
            # System prompt for query refinement
            system_prompt = """You are an expert at creating effective web search queries.
Your task is to refine user queries to make them more effective for retrieving
relevant information from web search engines.

When refining queries:
1. Preserve the original intent and topic of the query
2. Make the query more specific and targeted for search engines
3. Add terms that ensure results are current and up-to-date (like the current year)
4. Focus on retrieving factual, up-to-date information
5. Remove unnecessary words or fillers
6. If the query is pharmaceutical-related, add relevant terminology
7. Include 'latest' or 'current' for queries requiring recent information
8. DO NOT change the fundamental subject of the query
9. If the query is about tariffs, rates, or regulations, add terms for recency

Return ONLY the refined query without explanation or additional text."""

            # User prompt template
            user_prompt = f"Original query: {query}\n\nRefine this query to get better search results while preserving its original intent. Ensure the results will be current and up-to-date."

            # Generate refined query
            response = self.client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            refined_query = response.choices[0].message.content.strip()
            
            # Use refined query if it looks reasonable
            if refined_query and len(refined_query) > 5:
                logger.info(f"Refined query: '{refined_query}' (from: '{query}')")
                return refined_query
            else:
                logger.warning(f"Query refinement returned unusable result, using original")
                return query
                
        except Exception as e:
            logger.error(f"Error refining search query: {str(e)}")
            logger.error(traceback.format_exc())
            return query  # Fall back to original query on error
    
    def search(self, query: str, num_results: int = 8, refine_query: bool = True) -> Dict[str, Any]:
        """
        Search the web using SERPER API.
        
        Args:
            query: The search query
            num_results: Number of results to return
            refine_query: Whether to refine the query for better results
            
        Returns:
            Dict containing search results and status
        """
        try:
            logger.info(f"Performing web search for query: '{query}'")
            
            # Check cache first
            cache_key = query.lower().strip()
            if cache_key in self.query_cache:
                logger.info(f"Using cached results for query: '{query}'")
                return self.query_cache[cache_key]
            
            # Check if this is a weather/temperature query
            is_temperature_query = bool(re.search(r'\b(weather|temperature|forecast)\b', query, re.IGNORECASE))
            
            # Refine the query if requested (but skip refinement for temperature queries)
            if refine_query and not is_temperature_query:
                search_query = self.refine_search_query(query)
            else:
                search_query = query
            
            # For temperature queries, ensure we're asking directly
            if is_temperature_query:
                logger.info(f"Using direct temperature query: '{query}'")
            
            # Prepare the request payload
            payload = {
                'q': search_query,
                'num': num_results
            }
            
            # Make the API request
            response = requests.post(
                self.api_url, 
                headers=self.headers,
                json=payload
            )
            
            # Check for successful response
            if response.status_code == 200:
                search_results = response.json()
                
                # Process and format the results
                formatted_results = self._format_results(search_results)
                
                logger.info(f"Web search completed successfully, found {len(formatted_results)} results")
                
                # Check if we should store this result in tariff-specific format
                result_dict = {
                    'status': 'success',
                    'message': 'Web search completed successfully',
                    'results': formatted_results,
                    'raw_results': search_results,
                    'query': query,
                    'search_query': search_query
                }
                
                # Add to cache
                self.query_cache[cache_key] = result_dict
                
                # Manage cache size
                if len(self.query_cache) > self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                
                return result_dict
            else:
                logger.error(f"SERPER API error: {response.status_code} - {response.text}")
                return {
                    'status': 'error',
                    'message': f'SERPER API error: {response.status_code}',
                    'results': [],
                    'query': query
                }
                
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Error in web search: {str(e)}',
                'results': [],
                'query': query
            }
    
    def _format_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format raw search results into a standardized structure.
        
        Args:
            raw_results: Raw results from SERPER API
            
        Returns:
            List of formatted search results
        """
        formatted_results = []
        
        # Process answer box if available - prioritize this for weather and temperature queries
        if 'answerBox' in raw_results:
            ab = raw_results['answerBox']
            
            # For weather/temperature, add more detailed information
            is_weather = ('weather' in ab.get('title', '').lower() or 
                         'temperature' in ab.get('title', '').lower() or
                         'forecast' in ab.get('title', '').lower())
            
            formatted_result = {
                'title': ab.get('title', ''),
                'answer': ab.get('answer', ''),
                'snippet': ab.get('snippet', ''),
                'source': ab.get('source', ''),
                'sourceLink': ab.get('sourceLink', ''),
                'type': 'answer_box',
                'is_weather': is_weather
            }
            formatted_results.append(formatted_result)
        
        # Process knowledge graph if available
        if 'knowledgeGraph' in raw_results:
            kg = raw_results['knowledgeGraph']
            formatted_result = {
                'title': kg.get('title', ''),
                'description': kg.get('description', ''),
                'type': 'knowledge_graph'
            }
            formatted_results.append(formatted_result)
        
        # Process organic search results
        if 'organic' in raw_results:
            for result in raw_results['organic']:
                # Check the result has useful content
                if not result.get('title') or not result.get('snippet'):
                    continue
                    
                formatted_result = {
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'position': result.get('position', 0),
                    'type': 'organic',
                    'date': self._extract_date(result.get('date', ''), result.get('snippet', ''))
                }
                formatted_results.append(formatted_result)
            
        return formatted_results
    
    def _extract_date(self, date_str: str, snippet: str) -> str:
        """Extract and normalize a publication date from the result."""
        # If date is already provided by the API, use it
        if date_str:
            return date_str
            
        # Try to extract date from snippet
        date_patterns = [
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, snippet, re.IGNORECASE)
            if match:
                return match.group(1)
                
        # Default to empty string if no date found
        return ""
    
    def _format_web_results_for_tariffs(self, snippets, query):
        """Format web search results consistently for tariff and trade queries."""
        if not snippets:
            return "No relevant information found about current tariffs."
            
        formatted_results = "## Current Tariffs Information\n\n"
        formatted_results += "Based on web search results:\n\n"
        
        # Extract tariff rates and affected products
        tariff_rates = []
        affected_products = []
        recommendations = []
        sources = []
        dates = []
        
        for snippet in snippets:
            text = snippet.get('snippet', '')
            title = snippet.get('title', '')
            link = snippet.get('link', '')
            date = snippet.get('date', '')
            
            # Add date if available
            if date and date not in dates:
                dates.append(date)
                
            # Add source info
            if link and title:
                domain_match = re.search(r'https?://(?:www\.)?([^/]+)', link)
                domain = domain_match.group(1) if domain_match else "unknown source"
                sources.append({"title": title, "link": link, "domain": domain, "date": date})
            
            # Look for tariff percentages with specific pattern matching
            rate_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*(?:tariff|duty|tax)', text)
            for rate in rate_matches:
                tariff_rates.append(rate)
                
            # Also look for tariff amounts
            amount_matches = re.findall(r'(?:tariff|duty|tax)(?:\s+of)?\s+(\d+(?:\.\d+)?)\s*%', text)
            for rate in amount_matches:
                if rate not in tariff_rates:
                    tariff_rates.append(rate)
                
            # Look for affected product mentions
            if 'china' in query.lower():
                # Pharmaceutical specific product matching
                pharma_matches = re.findall(r'(?:pharmaceutical|medicine|drug|medical|api|healthcare)(?:[^.]*)', text.lower())
                for match in pharma_matches:
                    affected_products.append(match.strip())
                
                # General product matching
                if any(term in text.lower() for term in ['section 301', 'additional duty', 'list 1', 'list 2', 'list 3', 'list 4']):
                    lines = text.split('.')
                    for line in lines:
                        if any(term in line.lower() for term in ['product', 'goods', 'items', 'pharmaceuticals', 'medical']):
                            affected_products.append(line.strip())
            
            # Look for recommendations or actions
            if any(term in text.lower() for term in ['recommend', 'should', 'consider', 'advise']):
                lines = text.split('.')
                for line in lines:
                    if any(term in line.lower() for term in ['recommend', 'should', 'consider', 'advise']):
                        recommendations.append(line.strip())
        
        # Format dates information
        if dates:
            formatted_results += "### Information Currency\n"
            formatted_results += f"* Data based on sources dated {', '.join(dates[:3])}\n\n"
        
        # Format tariff information
        if tariff_rates:
            formatted_results += "### Tariff Rates\n"
            unique_rates = list(set(tariff_rates))
            for rate in unique_rates[:5]:  # Show more rates
                formatted_results += f"* {rate}% tariff mentioned in search results\n"
        else:
            formatted_results += "### Tariff Rates\n"
            formatted_results += "* Specific tariff rates were not identified in search results\n"
            
        # Format affected products
        if affected_products:
            formatted_results += "\n### Affected Products\n"
            # Use a set to deduplicate while preserving order of first occurrence
            seen = set()
            unique_products = [p for p in affected_products if not (p.lower() in seen or seen.add(p.lower()))]
            for product in unique_products[:5]:  # Show more products
                formatted_results += f"* {product}\n"
        else:
            formatted_results += "\n### Affected Products\n"
            formatted_results += "* Specific affected products were not identified in search results\n"
            
        # Format recommendations
        if recommendations:
            formatted_results += "\n### Recommendations\n"
            # Use a set to deduplicate while preserving order of first occurrence
            seen = set()
            unique_recommendations = [r for r in recommendations if not (r.lower() in seen or seen.add(r.lower()))]
            for rec in unique_recommendations[:5]:  # Show more recommendations
                formatted_results += f"* {rec}\n"
        else:
            formatted_results += "\n### Recommendations\n"
            formatted_results += "* Consider consulting official customs and trade resources for the most accurate and up-to-date information\n"
            formatted_results += "* Evaluate potential impacts on supply chain and pricing strategies\n"
            formatted_results += "* Review product classifications to determine tariff applicability\n"
            formatted_results += "* Monitor ongoing trade negotiations and policy changes\n"
        
        # Add sources section at the end
        if sources:
            formatted_results += "\n### Sources\n"
            for i, source in enumerate(sources[:5]):  # Show more sources
                date_info = f" ({source['date']})" if source['date'] else ""
                formatted_results += f"• {source['title']} - {source['domain']}{date_info}\n"
        
        return formatted_results
        
    def generate_web_context(self, query: str) -> List[Dict[str, str]]:
        """
        Generate web search context for a query
        
        Args:
            query: The query to search for
            
        Returns:
            List of search results with title, snippet, and url
        """
        try:
            logger.info(f"Generating web context for query: {query}")
            
            # Determine if this is a tariff query
            is_tariff_query = any(term in query.lower() for term in ['tariff', 'import duty', 'customs duty', 'export tax'])
            
            # Use more results for all factual queries to ensure we have enough data
            num_results = 10
            
            # Perform the search with comprehensive results
            search_results = self.search(query, num_results=num_results)
            
            if not search_results or search_results.get('status') != 'success' or not search_results.get('results'):
                logger.warning("No search results found")
                return []
            
            # Format results for context
            web_context = []
            # Get max results with fallback
            max_results = num_results
            logger.info(f"Using max_results={max_results} for web context")
            
            # First pass to identify numerical data and factual information
            numerical_data = self._extract_numerical_data(search_results.get('results', []), query)
            
            for i, result in enumerate(search_results.get('results', [])):
                if i >= max_results:
                    break
                    
                # Extract relevant information
                title = result.get('title', f"Search Result {i+1}")
                snippet = result.get('snippet', "No description available")
                url = result.get('link', "")
                
                # Add date information for recency assessment
                date = result.get('date', "")
                
                # Create a standardized result object
                formatted_result = {
                    'title': title,
                    'snippet': snippet,
                    'url': url,
                    'date': date
                }
                
                web_context.append(formatted_result)
            
            logger.info(f"Generated web context with {len(web_context)} results")
            
            # For tariff queries, add specialized tariff processing
            if is_tariff_query and web_context:
                # Extract and add tariff-specific information
                tariff_info = self._extract_tariff_info(web_context, query)
                if tariff_info:
                    # Add an additional entry with structured tariff information
                    web_context.append({
                        'title': "Structured Tariff Information Summary",
                        'snippet': tariff_info,
                        'url': "",
                        'date': datetime.now().strftime("%Y-%m-%d")
                    })
            
            # For all queries, add the factual data summary if available
            if numerical_data and len(numerical_data) > 0:
                # Create a factual data summary
                factual_info = self._format_numerical_data(numerical_data, query)
                if factual_info:
                    web_context.append({
                        'title': "Factual Data Summary",
                        'snippet': factual_info,
                        'url': "",
                        'date': datetime.now().strftime("%Y-%m-%d")
                    })
            
            return web_context
            
        except Exception as e:
            logger.error(f"Error generating web context: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _extract_numerical_data(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Extract numerical data and factual information from search results."""
        numerical_data = []
        
        try:
            # Define patterns to look for numbers in different contexts
            patterns = {
                "percentage": r'(\d+(?:\.\d+)?)\s*%',
                "currency": r'(?:USD|\$|EUR|€|GBP|£)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
                "quantity": r'(\d+(?:,\d+)*(?:\.\d+)?)\s+(?:units|items|pieces|packages|bottles|containers)',
                "date_value": r'(?:in|on|by|since|from)\s+(\d{4})',
                "duration": r'(\d+)\s+(?:days|weeks|months|years)',
                "weight": r'(\d+(?:\.\d+)?)\s+(?:kg|kilograms|g|grams|mg|lb|pounds|oz|ounces)',
                "range": r'(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)',
                "general_number": r'(\d+(?:,\d+)*(?:\.\d+)?)\s+(?:[a-zA-Z]+)'
            }
            
            # Define important fact indicators
            fact_indicators = [
                "according to", "reported", "estimated", "research shows", 
                "studies indicate", "data shows", "statistics", "survey",
                "analysis", "published", "report", "findings"
            ]
            
            for result in results:
                extract = {
                    "source": result.get('title', ''),
                    "numbers": [],
                    "facts": []
                }
                
                snippet = result.get('snippet', '')
                
                # Extract numbers based on patterns
                for context, pattern in patterns.items():
                    matches = re.findall(pattern, snippet)
                    for match in matches:
                        if isinstance(match, tuple):  # For patterns with multiple groups
                            for m in match:
                                if m:  # Non-empty matches
                                    # Look for surrounding context (10 words before and after)
                                    match_pos = snippet.find(m)
                                    if match_pos >= 0:
                                        start_pos = max(0, snippet.rfind('.', 0, match_pos))
                                        end_pos = snippet.find('.', match_pos)
                                        if end_pos < 0:
                                            end_pos = len(snippet)
                                        context_text = snippet[start_pos:end_pos].strip()
                                        
                                        extract["numbers"].append({
                                            "value": m,
                                            "context": context_text,
                                            "type": context
                                        })
                        else:
                            # Look for surrounding context
                            match_pos = snippet.find(match)
                            if match_pos >= 0:
                                start_pos = max(0, snippet.rfind('.', 0, match_pos))
                                end_pos = snippet.find('.', match_pos)
                                if end_pos < 0:
                                    end_pos = len(snippet)
                                context_text = snippet[start_pos:end_pos].strip()
                                
                                extract["numbers"].append({
                                    "value": match,
                                    "context": context_text,
                                    "type": context
                                })
                
                # Extract factual statements
                sentences = re.split(r'[.!?]', snippet)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if any(indicator in sentence.lower() for indicator in fact_indicators):
                        extract["facts"].append(sentence)
                
                # Only include if we found numbers or facts
                if extract["numbers"] or extract["facts"]:
                    numerical_data.append(extract)
            
            return numerical_data
            
        except Exception as e:
            logger.error(f"Error extracting numerical data: {str(e)}")
            return []
    
    def _format_numerical_data(self, numerical_data: List[Dict[str, Any]], query: str) -> str:
        """Format numerical data into a structured summary."""
        if not numerical_data:
            return ""
            
        formatted_result = "## Factual Data Summary\n\n"
        formatted_result += "Based on web search results, here are key numerical facts:\n\n"
        
        # Collect all numbers by type
        numbers_by_type = {}
        facts = []
        
        for data in numerical_data:
            source = data.get("source", "")
            
            # Process numbers
            for number in data.get("numbers", []):
                number_type = number.get("type", "general")
                if number_type not in numbers_by_type:
                    numbers_by_type[number_type] = []
                
                # Add source to the number entry
                number_entry = {
                    "value": number.get("value", ""),
                    "context": number.get("context", ""),
                    "source": source
                }
                numbers_by_type[number_type].append(number_entry)
            
            # Process facts
            for fact in data.get("facts", []):
                facts.append({
                    "statement": fact,
                    "source": source
                })
        
        # Group and format numbers by type
        for number_type, numbers in numbers_by_type.items():
            if numbers:
                formatted_result += f"### {number_type.title()} Data\n"
                
                # Deduplicate and limit to top 5 most relevant numbers
                seen_contexts = set()
                unique_numbers = []
                
                for number in numbers:
                    # Use a simplified version of context to detect duplicates
                    simplified_context = ' '.join(number["context"].lower().split()[:5])
                    if simplified_context not in seen_contexts:
                        seen_contexts.add(simplified_context)
                        unique_numbers.append(number)
                        
                        if len(unique_numbers) >= 5:
                            break
                
                for number in unique_numbers:
                    formatted_result += f"* {number['context']}\n"
                
                formatted_result += "\n"
        
        # Format factual statements
        if facts:
            formatted_result += "### Key Facts\n"
            
            # Deduplicate facts
            seen_facts = set()
            unique_facts = []
            
            for fact in facts:
                # Use a simplified version of the fact to detect duplicates
                simplified_fact = ' '.join(fact["statement"].lower().split()[:8])
                if simplified_fact not in seen_facts:
                    seen_facts.add(simplified_fact)
                    unique_facts.append(fact)
                    
                    if len(unique_facts) >= 5:
                        break
            
            for fact in unique_facts:
                formatted_result += f"* {fact['statement']}\n"
            
            formatted_result += "\n"
        
        # Add note about data quality
        formatted_result += "### Note\n"
        formatted_result += "* This data comes from web sources and may require verification\n"
        formatted_result += "* Facts and figures are extracted automatically and should be interpreted with caution\n"
        
        return formatted_result
    
    def _extract_tariff_info(self, results: List[Dict[str, str]], query: str) -> str:
        """Extract and structure tariff information from web results."""
        try:
            # Use the format method to structure the tariff information
            tariff_info = self._format_web_results_for_tariffs(results, query)
            return tariff_info
        except Exception as e:
            logger.error(f"Error extracting tariff information: {str(e)}")
            return "" 