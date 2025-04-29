import os
import json
import traceback
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import chromadb

logger = logging.getLogger(__name__)

class ChromaParser:
    """Parse and access ChromaDB collections."""
    
    def __init__(self, base_dir="vector_db_separate"):
        """Initialize the ChromaParser with the base directory."""
        self.base_dir = base_dir
        self.collection_data = {}
        self.clients = {}
        self.chroma_enabled = True
        
        # Map collection names to source names
        self.collection_to_source = {
            "inventory": "inventory",
            "transport": "transport",
            "guidelines": "guidelines",
            "policies": "policy"  # Note: policies collection maps to policy source
        }
        
        # Initialize collections
        if os.path.exists(base_dir):
            self._discover_collections()
        else:
            logger.warning(f"Base directory not found: {base_dir}")
            self.chroma_enabled = False
    
    def _discover_collections(self):
        """Discover all collections in the base directory."""
        try:
            # Get all subdirectories in the base directory
            collection_dirs = [d for d in os.listdir(self.base_dir) 
                              if os.path.isdir(os.path.join(self.base_dir, d))]
            
            logger.info(f"Found {len(collection_dirs)} potential collections in {self.base_dir}")
            
            # Initialize collection data
            for collection_dir in collection_dirs:
                try:
                    collection_path = os.path.join(self.base_dir, collection_dir)
                    metadata_path = os.path.join(collection_path, "metadata.json")
                    
                    # Default collection name is the directory name
                    collection_name = collection_dir
                    
                    # Try to read collection_name from metadata
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                if 'collection_name' in metadata:
                                    collection_name = metadata['collection_name']
                        except Exception as e:
                            logger.warning(f"Error reading metadata for {collection_dir}: {str(e)}")
                    
                    # Add to collection data
                    self.collection_data[collection_name] = {
                        'path': collection_path,
                        'db_path': os.path.join(collection_path, "chroma_db"),
                        'csv_path': os.path.join("embedded_data", f"{collection_name}_data_embedded.csv")
                    }
                    
                    logger.info(f"Added collection data for {collection_name}")
                    
                except Exception as e:
                    logger.error(f"Error discovering collection {collection_dir}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error discovering collections: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _create_client(self, collection_name: str) -> Optional[Any]:
        """Create a ChromaDB client for a collection with error handling."""
        if not self.chroma_enabled:
            return None
            
        if collection_name not in self.collection_data:
            logger.warning(f"Collection {collection_name} not found")
            return None
            
        try:
            collection_info = self.collection_data[collection_name]
            db_path = collection_info['db_path']
            
            # Check if ChromaDB directory exists
            if not os.path.exists(db_path):
                logger.warning(f"ChromaDB directory not found: {db_path}")
                return None
                
            logger.info(f"Attempting to create Chroma client for {collection_name} from {db_path}")
            
            # First try method: standard initialization
            try:
                client = chromadb.PersistentClient(path=db_path)
                collection = client.get_collection(name=collection_name)
                self.clients[collection_name] = client
                return collection
            except Exception as e:
                logger.warning(f"First initialization method failed for {collection_name}: {str(e)}")
                if "'_type'" not in str(e):  # If not a _type error, it might be another issue
                    raise  # Re-raise for unexpected errors
            
            # Second try method: get_or_create_collection
            try:
                client = chromadb.PersistentClient(path=db_path)
                collection = client.get_or_create_collection(name=collection_name)
                self.clients[collection_name] = client
                return collection
            except Exception as e:
                logger.warning(f"Second initialization method failed for {collection_name}: {str(e)}")
                raise  # Re-raise to fall back to CSV
                
        except Exception as e:
            logger.error(f"All initialization methods failed for {collection_name}")
            return None
    
    def get_collection_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        """Get all documents from a collection."""
        if not self.chroma_enabled:
            return self._get_csv_documents(collection_name)
            
        try:
            # Try to get client if not already created
            collection = None
            if collection_name in self.clients:
                try:
                    collection = self.clients[collection_name].get_collection(name=collection_name)
                except Exception as e:
                    logger.warning(f"Error getting cached collection {collection_name}: {str(e)}")
            
            if not collection:
                collection = self._create_client(collection_name)
            
            if not collection:
                logger.warning(f"Could not create ChromaDB client for {collection_name}, falling back to CSV")
                return self._get_csv_documents(collection_name)
                
            # Get all documents from the collection
            try:
                result = collection.get()
                
                # Process the results
                documents = []
                for i in range(len(result.get('ids', []))):
                    doc = {
                        'id': result['ids'][i] if 'ids' in result and i < len(result['ids']) else f"doc_{i}",
                        'content': result['documents'][i] if 'documents' in result and i < len(result['documents']) else "",
                        'metadata': result['metadatas'][i] if 'metadatas' in result and i < len(result['metadatas']) else {},
                        'source': self.collection_to_source.get(collection_name, collection_name)
                    }
                    documents.append(doc)
                
                logger.info(f"Got {len(documents)} documents from collection {collection_name}")
                return documents
                
            except Exception as e:
                logger.error(f"Error getting documents from collection {collection_name}: {str(e)}")
                return self._get_csv_documents(collection_name)
                
        except Exception as e:
            logger.error(f"Error in get_collection_documents for {collection_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_csv_documents(collection_name)
    
    def _get_csv_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        """Get documents from CSV fallback."""
        try:
            # Get CSV path from collection data or use default
            csv_path = None
            if collection_name in self.collection_data:
                csv_path = self.collection_data[collection_name].get('csv_path')
            
            if not csv_path:
                # Try standard naming convention
                source_name = self.collection_to_source.get(collection_name, collection_name)
                csv_path = f"embedded_data/{source_name}_data_embedded.csv"
            
            # Check if CSV exists
            if not os.path.exists(csv_path):
                logger.warning(f"CSV fallback not found: {csv_path}")
                return []
                
            logger.info(f"Reading documents from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Convert dataframe to documents
            documents = []
            for i, row in df.iterrows():
                doc = {
                    'id': row.get('id', str(i)),
                    'content': row.get('content', str(row)),
                    'metadata': {},
                    'source': self.collection_to_source.get(collection_name, collection_name)
                }
                
                # Add metadata from other columns
                for col in df.columns:
                    if col not in ['id', 'content', 'embedding', 'source']:
                        doc['metadata'][col] = row.get(col)
                
                documents.append(doc)
                
            logger.info(f"Got {len(documents)} documents from CSV for {collection_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting CSV documents for {collection_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def search_collection(self, collection_name: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search a collection with a query text."""
        if not self.chroma_enabled:
            logger.warning(f"ChromaDB is disabled, cannot search collection {collection_name}")
            return []
        
        try:
            # Try to get client if not already created
            collection = None
            if collection_name in self.clients:
                try:
                    collection = self.clients[collection_name].get_collection(name=collection_name)
                except Exception as e:
                    logger.warning(f"Error getting cached collection {collection_name}: {str(e)}")
            
            if not collection:
                collection = self._create_client(collection_name)
            
            if not collection:
                logger.warning(f"Could not create ChromaDB client for {collection_name}, cannot search")
                return []
                
            # Search the collection
            try:
                result = collection.query(
                    query_texts=[query_text],
                    n_results=top_k
                )
                
                # Process the results
                documents = []
                for i in range(len(result.get('ids', [[]])[0])):
                    doc = {
                        'id': result['ids'][0][i] if 'ids' in result and len(result['ids']) > 0 and i < len(result['ids'][0]) else f"doc_{i}",
                        'content': result['documents'][0][i] if 'documents' in result and len(result['documents']) > 0 and i < len(result['documents'][0]) else "",
                        'metadata': result['metadatas'][0][i] if 'metadatas' in result and len(result['metadatas']) > 0 and i < len(result['metadatas'][0]) else {},
                        'similarity': 1.0 - result['distances'][0][i] if 'distances' in result and len(result['distances']) > 0 and i < len(result['distances'][0]) else 0.0,
                        'source': self.collection_to_source.get(collection_name, collection_name)
                    }
                    documents.append(doc)
                
                logger.info(f"Found {len(documents)} results in collection {collection_name}")
                return documents
                
            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error in search_collection for {collection_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def get_dataframe(self, collection_name: str) -> pd.DataFrame:
        """Get a DataFrame representation of a collection."""
        try:
            # Get documents from collection
            documents = self.get_collection_documents(collection_name)
            
            if not documents:
                logger.warning(f"No documents found for collection {collection_name}")
                return pd.DataFrame()
        
            # Convert to dataframe
            rows = []
            for doc in documents:
                row = {
                    'content': doc.get('content', ''),
                    'source': doc.get('source', collection_name),
                    'id': doc.get('id', '')
                }
                
                # Add metadata to row
                metadata = doc.get('metadata', {})
                for key, value in metadata.items():
                    row[key] = value
                
                rows.append(row)
                
            df = pd.DataFrame(rows)
            logger.info(f"Created DataFrame with {len(df)} rows for collection {collection_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting DataFrame for {collection_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Get dataframes for all collections"""
        result = {}
        for collection_name in self.collection_data:
            source_name = self.collection_to_source.get(collection_name, collection_name)
            df = self.get_dataframe(collection_name)
            if not df.empty:
                result[source_name] = df
        
        return result
    
    def search_all_collections(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """Search all collections"""
        results = {}
        for collection_name in self.collection_data:
            source_name = self.collection_to_source.get(collection_name, collection_name)
            collection_results = self.search_collection(collection_name, query, top_k)
            if collection_results:
                results[source_name] = collection_results
        
        return results
    
    def get_all_dataframes_with_query(self, query: str, top_k: int = 5) -> Dict[str, pd.DataFrame]:
        """Get dataframes with search results for all collections"""
        results = self.search_all_collections(query, top_k)
        
        dataframes = {}
        for source_name, docs in results.items():
            if docs:
                df = pd.DataFrame(docs)
                df['embedding_array'] = None  # Add placeholder for compatibility
                dataframes[source_name] = df
        
        return dataframes 
