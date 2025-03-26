import os
import sys
from dotenv import load_dotenv

def init_system():
    """Initialize the system by creating necessary directories and checking dependencies"""
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = ['OPENAI_API_KEY', 'OPENAI_MODEL', 'EMBEDDING_MODEL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    # Create required directories
    directories = ['sessions', 'indexes']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Check data files
    data_files = [
        'embedded_data/inventory_data_embedded.csv',
        'embedded_data/transport_history_embedded.csv',
        'embedded_data/us_government_guidelines_for_medicine_transportation_and_storage_embedded.csv',
        'embedded_data/inventory_management_policy_embedded.csv'
    ]
    
    missing_files = [f for f in data_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing required data files: {', '.join(missing_files)}")
        sys.exit(1)
    
    print("System initialization completed successfully!")

if __name__ == '__main__':
    init_system() 