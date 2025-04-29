import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from config.settings import AppConfig
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import traceback

logger = logging.getLogger(__name__)

class RAGAgent:
    """Retrieval Augmented Generation agent for handling queries."""
    
    def __init__(self, config=None):
        """Initialize the RAG agent with configuration."""
        self.config = config or AppConfig()
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            logger.info("RAG Agent initialized with OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
            
        # Configure system prompt for better responses
        self.system_prompt = """You are an expert pharmaceutical inventory and supply chain assistant.
Your role is to provide accurate, detailed, and helpful information about pharmaceutical inventory,
transportation, regulations, and company policies based on the provided context.

CONSISTENCY REQUIREMENTS:
1. Always use the SAME structure and format for similar query types
2. For tariff/regulatory queries, always organize with clear section headings in this order:
   - Current Status (what is known)
   - Impact on Products (which items are affected)
   - Recommendations (what actions to take)
3. For inventory queries, always present:
   - Current inventory status
   - Key metrics (with exact numbers)
   - Any supply chain implications
4. Use the SAME title format (e.g., "Current Tariffs on Imports from China")
5. Always use dark green color for section headings

When responding:
1. Focus on the specific question and provide direct answers
2. Use data from the context to support your response
3. For inventory queries, include stock levels, expiry information, and reorder status
4. For transport queries, include shipping status, delays, and compliance factors
5. For regulatory queries, cite specific guidelines and requirements
6. When web search results are included, cite them appropriately and integrate the information
7. When appropriate, suggest actions based on industry best practices
8. When the full knowledgebase context is provided, synthesize information from across all data sources

RESPONSE STRUCTURE:
- Use a consistent hierarchical structure with the same heading levels for similar queries
- For similar/repeated query topics, maintain identical section titles and ordering
- Start with a title that clearly states the topic (e.g., "Current Tariffs on Imports from China")
- Organize all responses with bullet points for readability
- For complex topics, always use 3 main sections: Status/Information, Impact, and Recommendations

FORMATTING GUIDELINES:
- Use markdown formatting to enhance readability and organization
- Use appropriate headings that reflect the actual content and context
- Highlight important metrics and values using bold formatting
- Organize related information using lists when appropriate
- Format numbers consistently (with separators for large values)
- If presenting tabular data, use proper alignment
- Ensure visual spacing and separation between different content sections

If the context doesn't contain sufficient information, acknowledge the limitations
and provide the best possible answer with the available data.

Always aim for responses that are clear, accurate, and tailored specifically to the query content."""

        # Template for formatting the RAG query
        self.rag_template = """
Context Information:
{context}

User Query:
{query}

Provide a clear, concise response addressing the query based on the context information.
Include relevant facts, numbers, and insights from the context.
"""
        
        logger.info("RAG Agent initialization complete")
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _find_relevant_context(self, query: str, data_dict: Dict[str, pd.DataFrame], top_k: int = None) -> List[Dict]:
        """Find the most relevant context from data sources for a query."""
        try:
            # Use config value for top_k if not specified
            if top_k is None:
                top_k = self.config.TOP_K_RESULTS
                
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                logger.error("Failed to get query embedding")
                return []
            
            all_results = []
            
            # Process each data source
            for source_name, df in data_dict.items():
                # Skip empty dataframes
                if df.empty:
                    logger.warning(f"Empty dataframe for source {source_name}")
                    continue
                
                # Check if the dataframe already has similarity scores (from ChromaDB)
                if 'similarity' in df.columns and 'content' in df.columns:
                    # Data is already from ChromaDB, use the pre-calculated similarity scores
                    logger.info(f"Using pre-calculated similarity scores for {source_name}")
                    
                    # Sort by similarity if needed
                    if not df.empty:
                        df = df.sort_values('similarity', ascending=False)
                    
                    # Add results to the list
                    for idx, row in df.iterrows():
                        row_data = row.to_dict()
                        if 'source' not in row_data:
                            row_data['source'] = source_name
                        all_results.append(row_data)
                    
                    continue
                    
                # Check if dataframe has embedding_array column
                if 'embedding_array' not in df.columns and 'embedding' in df.columns:
                    # Convert embedding column to embedding_array
                    logger.info(f"Converting embedding string to array for {source_name}")
                    try:
                        df['embedding_array'] = df['embedding'].apply(
                            lambda x: np.array([float(i) for i in x.split(',')])
                        )
                    except Exception as e:
                        logger.error(f"Error converting embeddings: {str(e)}")
                        continue
                
                if 'embedding_array' not in df.columns:
                    logger.warning(f"No embedding or embedding_array column in {source_name} data")
                    continue
                
                # Calculate similarity scores
                similarities = []
                for idx, row in df.iterrows():
                    try:
                        embedding_array = row['embedding_array']
                        if not isinstance(embedding_array, np.ndarray):
                            logger.warning(f"Invalid embedding array at index {idx}")
                            continue
                            
                        sim = cosine_similarity(
                            query_embedding.reshape(1, -1),
                            embedding_array.reshape(1, -1)
                        )[0][0]
                        similarities.append((sim, idx, source_name))
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for row {idx}: {str(e)}")
                        continue
                
                # Add source's top results
                if similarities:
                    similarities.sort(reverse=True)
                    source_top_k = min(top_k, len(similarities))
                    
                    for i in range(source_top_k):
                        if i < len(similarities):
                            sim, idx, src = similarities[i]
                            row_data = df.iloc[idx].to_dict()
                            row_data['similarity'] = sim
                            row_data['source'] = src
                            all_results.append(row_data)
            
            # Sort all results by similarity score
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top results across all sources
            return all_results[:top_k]
        
        except Exception as e:
            logger.error(f"Error finding relevant context: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _format_context(self, relevant_items: List[Dict]) -> str:
        """Format context from relevant items for prompt."""
        context_parts = []
        
        for i, item in enumerate(relevant_items):
            # Extract item data
            content = item.get('content', '')
            source = item.get('source', 'unknown')
            metadata = item.get('metadata', {})
            
            # Skip empty content
            if not content:
                continue
            
            # Format metadata
            metadata_str = ""
            if metadata and isinstance(metadata, dict):
                metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items() if k not in ['content', 'source']])
            
            # Format the context item
            context_parts.append(f"[Source {i+1}: {source}]\n{content}\n{metadata_str}")
        
        # Join all context parts
        return "\n\n".join(context_parts)

    def _truncate_context(self, context: str, max_tokens: int = 8000) -> str:
        """Truncate context to fit within token limit."""
        # Simple approximate token count (can be improved with tiktoken)
        approx_tokens = len(context) / 4  # Rough estimate of 4 chars per token
        
        if approx_tokens <= max_tokens:
            return context
            
        # If too long, truncate to approximately max_tokens
        truncation_length = int(max_tokens * 4)
        truncated_context = context[:truncation_length]
        
        # Add indication that content was truncated
        truncated_context += "\n\n[Note: Some content was truncated due to length constraints]"
        
        logger.warning(f"Context truncated from {len(context)} chars to {len(truncated_context)} chars")
        return truncated_context

    def generate_response(self, query: str, data_dict: Dict[str, pd.DataFrame], 
                       additional_context: Optional[str] = None,
                       web_context: Optional[List[Dict[str, str]]] = None,
                       query_type: str = 'factual',
                       agent_type: str = 'vector_rag',
                       entities: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
        """Generate a response using RAG with the provided context from multiple data sources."""
        try:
            logger.info("Generating RAG response using available data sources")
            
            # Check if we have any context to work with
            if not data_dict or not any(df is not None and not df.empty for df in data_dict.values()):
                logger.warning("No valid data provided for response generation")
                if web_context and isinstance(web_context, list) and len(web_context) > 0:
                    logger.info("Using web context for response generation")
                    # Continue with web context only
                else:
                    return {
                        'status': 'error',
                        'response': "I couldn't find relevant information to answer your query. Could you please provide more details or rephrase your question?",
                        'message': "No valid data provided"
                    }
            
            # Calculate an appropriate max tokens based on context length
            max_context_length = self._estimate_total_context_length(data_dict, additional_context, web_context)
            if max_context_length > 6000:
                max_response_tokens = 1000
            else:
                max_response_tokens = 1500  # Default for normal-sized contexts
                
            logger.info(f"Context size estimate: {max_context_length}, setting max response tokens: {max_response_tokens}")
            
            # Build a prompt with the context from all sources
            prompt_parts = []
            
            # Get data sources from the data_dict keys
            data_sources = list(data_dict.keys())
            if web_context and isinstance(web_context, list) and len(web_context) > 0:
                data_sources.append('web')
            
            # Use router's _prepare_response_sources method to format sources nicely
            try:
                from agents.router import DataRouter
                router_instance = DataRouter(self.config)
                source_info = router_instance._prepare_response_sources(data_sources, data_dict, web_context)
            except Exception as e:
                logger.error(f"Error formatting sources: {str(e)}")
                # Fallback to basic source listing
                source_info = "Sources:\n" + "\n".join([f"• {source.capitalize()}" for source in data_sources])
            
            context = self._prepare_context(data_dict, additional_context, web_context)
            
            # Add query type and entities information to system prompt
            system_prompt = self._get_system_prompt(query_type, agent_type, entities)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}\n\nSources:\n{source_info}"}
            ]
            
            # Log the complete prompt for debugging
            logger.debug(f"Complete prompt: {json.dumps(messages, indent=2)}")
            
            # Get response from LLM
            # Get the model to use with fallback
            model = getattr(self.config, 'COMPLETION_MODEL', None) or self.config.OPENAI_MODEL
            logger.info(f"Using model {model} for response generation")
            
            response_text = self._get_completion(
                messages=messages,
                model=model,
                temperature=0.2,
                max_tokens=max_response_tokens
            )
            
            # Format response with citations if necessary
            formatted_response = self._format_response(response_text, query_type, agent_type)
            
            # Return successful response
            return {
                'status': 'success',
                'response': formatted_response,
                'sources': data_sources,
                'source_info': source_info
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'response': "I encountered an error while generating a response to your query. Please try again or rephrase your question.",
                'message': str(e)
            }
    
    def _get_system_prompt(self, query_type: str, agent_type: str, entities: Optional[Dict[str, List[str]]] = None) -> str:
        """Get the appropriate system prompt based on query type and agent type."""
        
        if query_type == 'analytical' or agent_type == 'analytics':
            return """You are an expert pharmaceutical inventory and supply chain analyst.
Your role is to provide detailed analytical responses about pharmaceutical inventory, 
transportation, regulations, and company policies based on the provided context.

When responding to analytical queries:
1. Focus on analyzing patterns, trends, and relationships in the data
2. Provide quantitative insights and metrics where appropriate
3. Organize information with clear section headings
4. Use bullet points for lists of findings or recommendations
5. For inventory queries, highlight important metrics like stock levels, turnover ratios
6. For transport queries, analyze delay patterns, cost factors, and compliance issues
7. For regulatory queries, explain implications and compliance requirements
8. When web search results are included, incorporate them into your analysis

Be thorough but concise, focusing on the most important insights.
Use a formal, precise, and analytical tone appropriate for pharmaceutical supply chain professionals."""

        elif query_type == 'hybrid' or agent_type == 'hybrid':
            return """You are an expert pharmaceutical compliance and operations analyst.
Your role is to provide responses that connect regulatory requirements with operational data
in the pharmaceutical supply chain.

When responding to hybrid queries:
1. First explain the relevant regulations or compliance requirements
2. Then analyze how the operational data relates to these requirements
3. Identify potential compliance issues or discrepancies
4. Suggest corrective actions or improvements where appropriate
5. Provide specific references to regulations when relevant
6. Connect the data points across different knowledge domains
7. When web search results are included, use them to supplement your analysis

Be precise, thorough, and objective in your assessment.
Use a formal and methodical tone that connects regulatory elements with operational data."""

        else:  # Default factual response
            return self.system_prompt  # Use the default prompt for factual queries
    
    def _get_content_prompt(self, query: str, context: str, 
                           query_type: str, agent_type: str,
                           entities: Optional[Dict[str, List[str]]] = None) -> str:
        """Get content prompt based on query type and entities."""
        
        # Base template
        base_template = """
Context Information:
{context}

User Query:
{query}
"""
        
        # Add special handling instructions based on query_type
        if query_type == 'analytical' or agent_type == 'analytics':
            analytical_instructions = """
Provide a clear analytical response addressing the query based on the context information.
Analyze the data to identify patterns, trends, and insights.
Include relevant metrics, quantities, and factual details from the context.
Organize your response with appropriate sections for better readability.
If web search results are provided, simply list the source details at the end of your response under a "Sources" section.
Do not include [Web 1], [Web 2] prefixes in your source citations.
"""
            return base_template.format(context=context, query=query) + analytical_instructions
            
        elif query_type == 'hybrid' or agent_type == 'hybrid':
            hybrid_instructions = """
Provide a response that connects regulatory requirements with the specific operational data.
First explain the relevant regulations or guidelines, then analyze the operational data in relation to these requirements.
Identify any compliance issues or areas of concern.
Suggest specific actions based on the findings.
If web search results are provided, simply list the source details at the end of your response under a "Sources" section.
Do not include [Web 1], [Web 2] prefixes in your source citations.
"""
            return base_template.format(context=context, query=query) + hybrid_instructions
            
        # For factual queries, add entity-specific instructions if available
        elif entities and (entities.get('medications') or entities.get('regulatory')):
            factual_with_entities = """
Provide a concise, direct response focusing on the specific {entity_type} mentioned in the query.
Include precise details about {entity_details} from the context.
Be direct and to the point while ensuring all relevant information is included.
If web search results are provided, simply list the source details at the end of your response under a "Sources" section.
Do not include [Web 1], [Web 2] prefixes in your source citations.
"""
            # Get the main entity type and details
            entity_type = "medications" if entities.get('medications') else "regulations"
            entity_details = ", ".join(entities.get(entity_type, []))
            
            return base_template.format(context=context, query=query) + \
                factual_with_entities.format(entity_type=entity_type, entity_details=entity_details)
        
        # Default factual prompt
        return base_template.format(context=context, query=query) + """
Provide a clear, concise response addressing the query based on the context information.
Include relevant facts, numbers, and insights from the context.
If web search results are provided, simply list the source details at the end of your response under a "Sources" section.
Do not include [Web 1], [Web 2] prefixes in your source citations.
"""
    
    def _add_citations(self, response_text: str, relevant_items: List[Dict]) -> str:
        """Add citations to the response for analytical queries."""
        # Only add citations if response is long enough to warrant them
        if len(response_text) < 200:
            return response_text
            
        # Check if response already has a Sources section
        if "### Sources" in response_text or "## Sources" in response_text:
            logger.info("Response already has Sources section, skipping citation addition")
            return response_text
            
        # Extract source information
        citations = {}
        for idx, item in enumerate(relevant_items[:5]):  # Limit to top 5 sources
            source = item.get('source', 'unknown')
            if source not in citations:
                citations[source] = []
                
            # Extract a brief reference
            content = item.get('content', '')
            if len(content) > 50:
                content = content[:50] + "..."
                
            citations[source].append(f"[{source.capitalize()} {idx+1}] {content}")
        
        # Add citations section if we have any
        if citations:
            response_text += "\n\n### Sources:\n"
            for source, refs in citations.items():
                response_text += f"\n{source.capitalize()}:\n"
                for ref in refs:
                    response_text += f"* {ref}\n"
                    
        return response_text

    def _prepare_context(self, data_dict, additional_context, web_context):
        """Prepare context from all data sources."""
        context_parts = []
        
        # Process data from different sources
        for source_name, df in data_dict.items():
            if df is not None and not df.empty:
                source_context = f"### {source_name.capitalize()} Data:\n"
                
                # For ChromaDB content-based data
                if 'content' in df.columns:
                    for idx, row in df.iterrows():
                        content = row.get('content', '')
                        if content:
                            source_context += f"{content}\n\n"
                # For structured data
                else:
                    # Convert DataFrame to formatted text
                    try:
                        # Get first few rows as string representation
                        max_rows = min(5, len(df))
                        source_context += f"{df.head(max_rows).to_string()}\n\n"
                    except Exception as e:
                        logger.warning(f"Error formatting {source_name} data: {str(e)}")
                        source_context += f"[Data available but could not be formatted]\n\n"
                
                context_parts.append(source_context)
        
        # Add additional context if provided
        if additional_context:
            context_parts.append(f"### Additional Context:\n{additional_context}")
        
        # Add web search results if provided
        if web_context and isinstance(web_context, list) and len(web_context) > 0:
            web_context_text = "### Web Search Results:\n"
            for idx, result in enumerate(web_context):
                if 'title' in result and 'snippet' in result:
                    web_context_text += f"{result['title']}\n{result['snippet']}\n\n"
            context_parts.append(web_context_text)
        
        # Join all context parts
        return "\n".join(context_parts)
        
    def _estimate_total_context_length(self, data_dict, additional_context, web_context):
        """Estimate the total context length to determine appropriate token limits."""
        total_length = 0
        
        # Estimate length from data sources
        for source_name, df in data_dict.items():
            if df is not None and not df.empty:
                # For ChromaDB content-based data
                if 'content' in df.columns:
                    for idx, row in df.iterrows():
                        content = row.get('content', '')
                        if content:
                            total_length += len(content)
                # For structured data
                else:
                    # Rough estimate based on DataFrame size
                    total_length += df.shape[0] * df.shape[1] * 10  # Rough average of 10 chars per cell
        
        # Add length of additional context
        if additional_context:
            total_length += len(additional_context)
        
        # Add length of web search results
        if web_context and isinstance(web_context, list):
            for result in web_context:
                if 'title' in result:
                    total_length += len(result['title'])
                if 'snippet' in result:
                    total_length += len(result['snippet'])
        
        logger.info(f"Estimated total context length: {total_length} characters")
        return total_length
        
    def _format_response(self, response_text, query_type, agent_type):
        """Format the response based on query type and agent type."""
        # If response already has good formatting, return as is
        if "##" in response_text or "**" in response_text:
            return response_text
            
        # For analytical queries, add some structure
        if query_type == 'analytical' or agent_type == 'analytics':
            # Check if response has natural sections
            if not any(marker in response_text for marker in [":", "-", "•", "1.", "*"]):
                # Add basic formatting with bullets for readability
                lines = response_text.split('\n')
                formatted_lines = []
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        formatted_lines.append(line)
                        continue
                        
                    # First line becomes a heading
                    if i == 0:
                        formatted_lines.append(f"## {line}")
                    # Other non-empty lines become bullet points
                    elif len(line) > 20:  # Only format substantial lines
                        formatted_lines.append(f"• {line}")
                    else:
                        formatted_lines.append(line)
                
                return "\n\n".join(formatted_lines)
        
        # For most responses, return as is
        return response_text
    
    def _get_completion(self, messages, model, temperature=0.2, max_tokens=1000):
        """Get completion from OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try again with reduced context if we hit token limits
            if "maximum context length" in str(e) and len(messages) > 1:
                logger.info("Retrying with reduced context due to token limits")
                # Get the system message and the last user message
                reduced_messages = [
                    messages[0],  # System message
                    {"role": "user", "content": "Please provide a shorter summary based on the available information."}
                ]
                
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=reduced_messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content.strip()
                except Exception as retry_error:
                    logger.error(f"Error in retry with reduced context: {str(retry_error)}")
            
            return "I apologize, but I encountered an error processing your request. Please try a more specific or shorter query." 