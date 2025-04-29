from datetime import datetime, timedelta
import json
import os
import logging
import time
from typing import Union, Dict, Any, List, Optional
import traceback

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages user sessions and chat history."""
    
    def __init__(self, session_dir: str = 'sessions'):
        """Initialize session manager with specified session directory."""
        self.session_dir = session_dir
        self.sessions = {}  # In-memory cache of loaded sessions
        os.makedirs(self.session_dir, exist_ok=True)
        logger.info(f"Session manager initialized with directory: {session_dir}")
    
    def _load_session(self, session_id: str) -> None:
        """
        Load a session from disk into memory.
        
        Args:
            session_id: Unique session identifier
        """
        try:
            if session_id in self.sessions:
                # Session already loaded
                return
                
            session_file = os.path.join(self.session_dir, f"{session_id}.json")
            
            if os.path.exists(session_file):
                try:
                    with open(session_file, 'r') as f:
                        self.sessions[session_id] = json.load(f)
                    logger.debug(f"Loaded session {session_id} from disk")
                except json.JSONDecodeError:
                    logger.error(f"Error parsing session file: {session_file}")
                    self.sessions[session_id] = {
                        'history': [],
                        'metadata': {},
                        'created_at': datetime.now().isoformat()
                    }
            else:
                # Initialize new session
                self.sessions[session_id] = {
                    'history': [],
                    'metadata': {},
                    'created_at': datetime.now().isoformat()
                }
                logger.debug(f"Initialized new session {session_id}")
                
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}")
            logger.error(traceback.format_exc())
            # Initialize empty session as fallback
            self.sessions[session_id] = {
                'history': [],
                'metadata': {},
                'created_at': datetime.now().isoformat()
            }
    
    def _save_session(self, session_id: str) -> None:
        """
        Save a session from memory to disk.
        
        Args:
            session_id: Unique session identifier
        """
        try:
            if session_id not in self.sessions:
                logger.error(f"Cannot save non-existent session: {session_id}")
                return
                
            session_file = os.path.join(self.session_dir, f"{session_id}.json")
            
            with open(session_file, 'w') as f:
                json.dump(self.sessions[session_id], f, indent=2)
            
            logger.debug(f"Saved session {session_id} to disk")
                
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {str(e)}")
            logger.error(traceback.format_exc())
    
    def add_to_history(self, session_id: str, message: Dict[str, Any]) -> None:
        """Add a message to session history."""
        if not session_id or not isinstance(message, dict):
            return
            
        self._load_session(session_id)
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': [],
                'metadata': {},
                'created_at': datetime.now().isoformat()
            }
            
        self.sessions[session_id]['history'].append(message)
        self.sessions[session_id]['last_updated'] = datetime.now().isoformat()
        
        self._save_session(session_id)
    
    def add_interaction(self, session_id: str, query: str, enhanced_query: str, 
                       response: str, sources: List[str], query_type: str) -> None:
        """
        Add a complete interaction to the session history with enriched metadata.
        
        Args:
            session_id: Session identifier
            query: Original user query
            enhanced_query: Enhanced query after processing
            response: Generated response
            sources: Data sources used
            query_type: Type of query (factual, analytical, hybrid)
        """
        if not session_id:
            return
            
        self._load_session(session_id)
        
        # Initialize session if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': [],
                'metadata': {
                    'query_types': {},
                    'sources_used': {},
                    'total_interactions': 0
                },
                'created_at': datetime.now().isoformat()
            }
        
        # Create interaction object
        timestamp = datetime.now().isoformat()
        interaction = {
            'query': query,
            'enhanced_query': enhanced_query,
            'response': response,
            'sources': sources,
            'query_type': query_type,
            'timestamp': timestamp
        }
        
        # Add to history
        self.sessions[session_id]['history'].append(interaction)
        
        # Update metadata
        self.sessions[session_id]['last_updated'] = timestamp
        
        # Update query type statistics
        query_types = self.sessions[session_id]['metadata'].get('query_types', {})
        query_types[query_type] = query_types.get(query_type, 0) + 1
        self.sessions[session_id]['metadata']['query_types'] = query_types
        
        # Update sources statistics
        sources_used = self.sessions[session_id]['metadata'].get('sources_used', {})
        for source in sources:
            sources_used[source] = sources_used.get(source, 0) + 1
        self.sessions[session_id]['metadata']['sources_used'] = sources_used
        
        # Update total interactions
        self.sessions[session_id]['metadata']['total_interactions'] = len(self.sessions[session_id]['history'])
        
        # Save session
        self._save_session(session_id)
    
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get chat history for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            List[Dict[str, Any]]: List of chat entries
        """
        try:
            # Validate session ID
            if not session_id or not isinstance(session_id, str):
                logger.error(f"Invalid session ID: {session_id}")
                return []
            
            # Load session if not already loaded
            self._load_session(session_id)
            
            # Return history from memory
            if session_id in self.sessions and 'history' in self.sessions[session_id]:
                history = self.sessions[session_id]['history']
                logger.info(f"Retrieved history for session {session_id}: {len(history)} entries")
                return history
            
            return []
                
        except Exception as e:
            logger.error(f"Error getting history for session {session_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def clear_history(self, session_id: str) -> bool:
        """
        Clear chat history for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            bool: Success status
        """
        try:
            # Validate session ID
            if not session_id or not isinstance(session_id, str):
                logger.error(f"Invalid session ID: {session_id}")
                return False
            
            # Create session file path
            session_file = os.path.join(self.session_dir, f"{session_id}.json")
            
            # Check if session file exists
            if not os.path.exists(session_file):
                logger.info(f"No history found for session {session_id}")
                return True  # Nothing to clear
            
            # Remove session file
            os.remove(session_file)
            
            # Remove from memory cache if loaded
            if session_id in self.sessions:
                del self.sessions[session_id]
                
            logger.info(f"Cleared history for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing history for session {session_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def get_all_sessions(self) -> List[str]:
        """
        Get a list of all session IDs.
        
        Returns:
            List[str]: List of session IDs
        """
        try:
            # Get all JSON files in session directory
            session_files = [f for f in os.listdir(self.session_dir) if f.endswith('.json')]
            
            # Extract session IDs from filenames
            session_ids = [os.path.splitext(f)[0] for f in session_files]
            
            logger.info(f"Found {len(session_ids)} sessions")
            return session_ids
            
        except Exception as e:
            logger.error(f"Error getting all sessions: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """
        Clean up old session files.
        
        Args:
            max_age_days: Maximum age of session files in days
            
        Returns:
            int: Number of sessions cleaned up
        """
        try:
            # Get current time
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            # Get all JSON files in session directory
            session_files = [os.path.join(self.session_dir, f) for f in os.listdir(self.session_dir) 
                            if f.endswith('.json')]
            
            # Check file age and remove old files
            count = 0
            for file_path in session_files:
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        
                        # Remove from memory cache if loaded
                        session_id = os.path.splitext(os.path.basename(file_path))[0]
                        if session_id in self.sessions:
                            del self.sessions[session_id]
                            
                        count += 1
                except Exception as e:
                    logger.error(f"Error cleaning up file {file_path}: {str(e)}")
            
            logger.info(f"Cleaned up {count} old sessions")
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {str(e)}")
            logger.error(traceback.format_exc())
            return 0 