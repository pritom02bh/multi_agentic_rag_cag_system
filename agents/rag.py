import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import FAISS with graceful fallback
try:
    import faiss.swigfaiss_avx2 as faiss
    logger.info("Successfully loaded FAISS with AVX2 support")
except ImportError:
    try:
        import faiss
        logger.info("Loaded FAISS without AVX2 support - this may impact performance but is still functional")
    except ImportError:
        logger.error("Failed to import FAISS - please ensure it is installed correctly")
        raise

from config.settings import AppConfig
import os
import json
from .analyzer import AnalyticalAgent

class RAGAgent:
    def __init__(self):
        self.config = AppConfig()
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=self.config.EMBEDDING_MODEL
        )
        self.llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=self.config.OPENAI_MODEL,
            temperature=0.7
        )
        # Enhanced prompt templates for different types of queries
        self.inventory_prompt = """
            You are an expert pharmaceutical inventory analyst and supply chain professional.
            Your role is to provide clear, insightful, and actionable analysis in a friendly yet professional tone.
            
            When analyzing inventory data, consider:
            1. Current stock levels and their implications
            2. Supply chain risks and opportunities
            3. Cost optimization opportunities
            4. Regulatory compliance aspects
            5. Patient care impact
            
            Format your response in a clear, structured manner using:
            - Professional greetings and natural transitions
            - Clear section headings when appropriate
            - Bullet points for key insights
            - Tables for comparative data
            - Specific recommendations with rationale
            
            Context information:
            {context}
            
            Question: {query}
            
            Provide a comprehensive analysis that is both professional and engaging. Use natural language and avoid overly technical jargon unless necessary.
            Include specific numbers and percentages when relevant, and always provide actionable recommendations.
        """
        
        self.transport_prompt = """
            You are a logistics and transportation analyst specializing in pharmaceutical supply chains.
            Your role is to analyze transport history data and provide actionable insights in a clear and professional manner.
            
            When analyzing transport data, consider:
            1. Delivery performance metrics
            2. Route efficiency and optimization
            3. Temperature control compliance
            4. Cost analysis and optimization
            5. Risk assessment and mitigation
            
            Format your response in a clear, structured manner using:
            - Professional greetings and natural transitions
            - Clear section headings when appropriate
            - Bullet points for key findings
            - Tables for comparative data
            - Specific recommendations with rationale
            
            Context information:
            {context}
            
            Question: {query}
            
            Provide a comprehensive analysis that is both professional and engaging. Focus on patterns, trends, and opportunities for improvement.
            Include specific metrics and percentages where relevant, and always provide actionable recommendations.
        """
        
        self.inventory_template = ChatPromptTemplate.from_template(self.inventory_prompt)
        self.transport_template = ChatPromptTemplate.from_template(self.transport_prompt)
        self.indexes = {}
        self.data = {}
        self._load_data()
        
    def _load_data(self):
        """Load and prepare data sources and FAISS indexes"""
        for source_key, file_path in self.config.DATA_SOURCES.items():
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Convert string embeddings back to numpy arrays
            embeddings = np.array([
                np.array([float(x) for x in emb.split(',')])
                for emb in df['embedding']
            ])
            
            # Create FAISS index
            dimension = embeddings.shape[1]  # Should be 1536 for text-embedding-3-small
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))
            
            # Store index and data
            self.indexes[source_key] = index
            self.data[source_key] = df
            
    def _get_relevant_documents(self, query: str, source: str, k: int = None) -> list:
        """
        Retrieve relevant documents using FAISS similarity search.
        
        Args:
            query (str): The query to search for
            source (str): The data source to search in
            k (int): Number of documents to retrieve
            
        Returns:
            list: List of relevant documents
        """
        if k is None:
            k = 50  # Increased from default to ensure we get all items
            
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Perform similarity search
        D, I = self.indexes[source].search(
            np.array([query_embedding]).astype('float32'), 
            k
        )
        
        # Get the relevant documents
        relevant_docs = self.data[source].iloc[I[0]]['content'].tolist()
        
        return relevant_docs
        
    def process(self, query: str, source: str) -> str:
        """
        Process a query using RAG to generate a response.
        Handles both inventory and transport history queries.
        
        Args:
            query (str): The enhanced query
            source (str): The selected data source
            
        Returns:
            str: Formatted response with proper structure
        """
        try:
            # Get relevant documents
            relevant_docs = self._get_relevant_documents(query, source)
            
            # Combine documents into context
            context = "\n\n".join(relevant_docs)
            
            # Determine query type and use appropriate prompt
            is_transport_query = any(keyword in query.lower() for keyword in [
                'transport', 'shipping', 'delivery', 'route', 'transit',
                'temperature', 'carrier', 'shipment'
            ])
            
            # Select appropriate prompt template
            prompt = self.transport_template if is_transport_query else self.inventory_template
            
            # Generate initial response using LLM
            chain = prompt | self.llm
            initial_response = chain.invoke({
                "context": context,
                "query": query
            })
            
            # Use the analytical agent to enhance the response
            analyzer = AnalyticalAgent()
            analysis_result = analyzer.analyze(initial_response.content)
            
            # Format the response in a structured way
            formatted_text = analysis_result.get('formatted_response', initial_response.content)
            insights = analysis_result.get('insights', '')
            charts = analysis_result.get('charts', [])
            
            # Create a properly formatted response with clean structure
            formatted_response = []
            
            # Add main content with proper formatting
            formatted_response.append(formatted_text)
            
            # Add insights section if available
            if insights:
                formatted_response.append("\n## Key Insights and Recommendations")
                formatted_response.append(insights)
            
            # Add visualization section if available
            if charts:
                formatted_response.append(f"\n## Visualizations")
                for i, chart in enumerate(charts, 1):
                    formatted_response.append(f"• {chart.get('title', 'Untitled Chart')} - {chart.get('description', 'Shows data analysis')}")
            
            return "\n".join(formatted_response)
            
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}")
            error_msg = (
                "An error occurred while processing your request.\n"
                f"Error details: {str(e)}\n"
                "Please try rephrasing your question or contact support if the issue persists."
            )
            return error_msg 