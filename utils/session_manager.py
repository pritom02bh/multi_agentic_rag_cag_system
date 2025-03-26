from datetime import datetime, timedelta
import json
import os

class SessionManager:
    def __init__(self):
        self.sessions_dir = 'sessions'
        if not os.path.exists(self.sessions_dir):
            os.makedirs(self.sessions_dir)
    
    def _get_session_file(self, session_id):
        return os.path.join(self.sessions_dir, f'{session_id}.json')
    
    def add_to_history(self, session_id, data):
        file_path = self._get_session_file(session_id)
        history = self.get_history(session_id)
        
        # Add timestamp to the data
        data['timestamp'] = datetime.now().isoformat()
        history.append(data)
        
        # Save updated history
        with open(file_path, 'w') as f:
            json.dump(history, f)
    
    def get_history(self, session_id):
        file_path = self._get_session_file(session_id)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return []
    
    def clear_history(self, session_id):
        file_path = self._get_session_file(session_id)
        if os.path.exists(file_path):
            os.remove(file_path)
    
    def cleanup_old_sessions(self, max_age_hours=24):
        """Remove session files older than max_age_hours"""
        now = datetime.now()
        for filename in os.listdir(self.sessions_dir):
            file_path = os.path.join(self.sessions_dir, filename)
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_modified > timedelta(hours=max_age_hours):
                os.remove(file_path) 