import os
import logging
import pandas as pd
import numpy as np
import traceback
from typing import Dict, List, Optional, Any
from config.settings import AppConfig
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import json
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Manages ChromaDB clients and collections."""
    
    def __init__(self, config=None):
        """Initialize the ChromaDB manager with configuration."""
        self.config = config or AppConfig()
        self.clients = {}
        self.persistent_clients = {}
        self.chroma_enabled = True  # Flag to control ChromaDB usage
        
        # Check if we should disable ChromaDB based on environment or config
        if hasattr(self.config, 'USE_CHROMA_DB') and not self.config.USE_CHROMA_DB:
            logger.info("ChromaDB usage is disabled via configuration")
            self.chroma_enabled = False
            return
        
        # Initialize embedding function if using OpenAI
        try:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.config.OPENAI_API_KEY,
                model_name=getattr(self.config, 'EMBEDDING_MODEL', 'text-embedding-3-small')
            )
            logger.info(f"Initialized OpenAI embeddings with model: {getattr(self.config, 'EMBEDDING_MODEL', 'text-embedding-3-small')}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embedding function: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning("ChromaDB will be disabled due to embedding function initialization failure")
            self.chroma_enabled = False
            self.embedding_function = None
            return
    
    def create_client(self, collection_name, base_path=None):
        """
        Create a ChromaDB client for a collection with error handling and retries.
        
        Args:
            collection_name: The name of the collection
            base_path: The base path to the collection directory
            
        Returns:
            ChromaDB Collection object or None if failed
        """
        if not self.chroma_enabled:
            logger.info(f"ChromaDB is disabled, skipping creation of client for {collection_name}")
            return None
            
        try:
            # Determine the path to the collection
            if base_path is None:
                base_path = os.path.join("vector_db_separate", collection_name)
            
            # Full path to the ChromaDB directory
            chroma_path = os.path.join(base_path, "chroma_db")
            
            # Check if the directory exists
            if not os.path.exists(chroma_path):
                logger.warning(f"ChromaDB directory not found for {collection_name}: {chroma_path}")
                return None
            
            # Try to get collection name from metadata
            metadata_file = os.path.join(base_path, "metadata.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        if 'collection_name' in metadata:
                            collection_name = metadata['collection_name']
                            logger.info(f"Using collection name from metadata: {collection_name}")
                except Exception as e:
                    logger.warning(f"Error reading metadata for {collection_name}: {str(e)}")
            
            # Create a persistent client for this collection
            logger.info(f"Attempting to create Chroma client for {collection_name} from {chroma_path}")
            
            # First attempt: Try with standard initialization
            try:
                client = chromadb.PersistentClient(path=chroma_path)
                collection = client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                self.persistent_clients[collection_name] = client
                self.clients[collection_name] = collection
                logger.info(f"Successfully initialized collection {collection_name} (Method 1)")
                return collection
            except Exception as e:
                logger.warning(f"First initialization method failed for {collection_name}: {str(e)}")
                # Don't crash on '_type' errors - these are common with version mismatches
                
            # Second attempt: Try with a different initialization method
            try:
                # Try different parameters for compatibility with older versions
                client = chromadb.PersistentClient(path=chroma_path)
                collection = client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                self.persistent_clients[collection_name] = client
                self.clients[collection_name] = collection
                logger.info(f"Successfully initialized collection {collection_name} (Method 2)")
                return collection
            except Exception as e:
                logger.warning(f"Second initialization method failed for {collection_name}: {str(e)}")
                
            # Final attempt: Use backup CSV data if available
            csv_path = os.path.join("embedded_data", f"{collection_name}_data_embedded.csv")
            if os.path.exists(csv_path):
                logger.info(f"ChromaDB initialization failed, will use CSV data from {csv_path}")
                # We'll return None here and let the calling code fall back to CSV
                
            logger.error(f"All initialization methods failed for {collection_name}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error creating ChromaDB client for {collection_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def get_collection(self, collection_name):
        """Get the collection by name."""
        if not self.chroma_enabled:
            return None
            
        if collection_name in self.clients:
            return self.clients[collection_name]
            
        # Try to create the client if it doesn't exist
        return self.create_client(collection_name)
    
    def search_collection(self, collection_name, query_text, top_k=5):
        """
        Search a collection with a query text.
        
        Args:
            collection_name: The name of the collection to search
            query_text: The query text
            top_k: Number of results to return
            
        Returns:
            List of results or empty list if error
        """
        if not self.chroma_enabled:
            logger.info(f"ChromaDB is disabled, skipping search in {collection_name}")
            return []
            
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                logger.warning(f"Collection {collection_name} not found or failed to initialize")
                return []
                
            # Search the collection
            results = collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            
            # Process the results
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            
            # Combine the results into a list of dictionaries
            processed_results = []
            for i in range(len(documents)):
                item = {
                    'content': documents[i],
                    'metadata': metadatas[i] if i < len(metadatas) else {},
                    'similarity': 1.0 - distances[i] if i < len(distances) else 0.0,
                    'source': collection_name
                }
                processed_results.append(item)
                
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def initialize_all_collections(self, base_dir="vector_db_separate"):
        """
        Initialize all collections in the base directory.
        
        Args:
            base_dir: The base directory containing collection subdirectories
            
        Returns:
            Dictionary mapping collection names to ChromaDB Collection objects
        """
        if not self.chroma_enabled:
            logger.info("ChromaDB is disabled, skipping collection initialization")
            return {}
            
        try:
            # Get all subdirectories in the base directory
            if not os.path.exists(base_dir):
                logger.warning(f"Base directory {base_dir} does not exist")
                return {}
                
            # Find all collection directories
            collection_dirs = [d for d in os.listdir(base_dir) 
                              if os.path.isdir(os.path.join(base_dir, d))]
            
            # Initialize each collection
            initialized_count = 0
            for collection_name in collection_dirs:
                collection = self.create_client(collection_name, os.path.join(base_dir, collection_name))
                if collection:
                    initialized_count += 1
            
            logger.info(f"Initialized {initialized_count} Chroma clients")
            
            # If no collections were initialized successfully, disable ChromaDB
            if initialized_count == 0 and len(collection_dirs) > 0:
                logger.warning("No ChromaDB clients were initialized successfully. Will fall back to CSV data.")
                self.chroma_enabled = False
                
            return self.clients
            
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")
            logger.error(traceback.format_exc())
            self.chroma_enabled = False
            return {}
    
    def query_collection(self, collection_name: str, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Query a specific collection
        
        Args:
            collection_name: Name of the collection to query
            query_text: The query text
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with query results
        """
        try:
            if collection_name not in self.clients:
                logger.error(f"Collection {collection_name} not initialized")
                return []
            
            collection = self.clients[collection_name]
            
            # Use similarity_search_with_score for retrieving docs with scores
            results = collection.similarity_search_with_score(
                query=query_text,
                k=top_k
            )
            
            # Process results
            processed_results = []
            for doc, score in results:
                # Convert the distance score to a similarity score (1 - distance)
                similarity = 1 - score if score <= 1 else 0
                
                # Create result dict
                processed_result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity': similarity,
                    'source': collection_name
                }
                processed_results.append(processed_result)
            
            logger.info(f"Found {len(processed_results)} results for {collection_name}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error querying collection {collection_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def query_all_collections(self, query_text: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Query all collections
        
        Args:
            query_text: The query text
            top_k: Number of results to return per collection
            
        Returns:
            Dictionary mapping collection names to lists of results
        """
        results = {}
        for collection_name in self.clients:
            collection_results = self.query_collection(collection_name, query_text, top_k)
            results[collection_name] = collection_results
        
        return results
    
    def convert_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """
        Convert query results to a DataFrame for compatibility with existing code
        
        Args:
            results: List of result dictionaries
            
        Returns:
            DataFrame with results
        """
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Add embedding_array column (empty for now, will be populated when needed)
        df['embedding_array'] = None
        
        return df
    
    def get_dataframes_dict(self, query_text: str, top_k: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Get results from all collections as DataFrames
        
        Args:
            query_text: The query text
            top_k: Number of results to return per collection
            
        Returns:
            Dictionary mapping source names to DataFrames
        """
        collection_results = self.query_all_collections(query_text, top_k)
        
        # Convert results to DataFrames
        dataframes_dict = {}
        for source_name, results in collection_results.items():
            dataframes_dict[source_name] = self.convert_to_dataframe(results)
        
        return dataframes_dict
        
    def search_collection_as_dataframe(self, collection_name: str, query_text: str, top_k: int = 5) -> pd.DataFrame:
        """
        Search a collection and return results as a DataFrame
        
        Args:
            collection_name: Name of the collection to query
            query_text: The query text
            top_k: Number of results to return
            
        Returns:
            DataFrame with search results
        """
        try:
            # Get results as a list of dictionaries
            results = self.query_collection(collection_name, query_text, top_k)
            
            # Convert to DataFrame
            return self.convert_to_dataframe(results)
        except Exception as e:
            logger.error(f"Error in search_collection_as_dataframe for {collection_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame() 