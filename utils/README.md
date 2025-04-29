# ChromaParser

A custom parser for ChromaDB collections without using the ChromaDB library. This parser allows direct access to the ChromaDB SQLite database to retrieve documents and metadata.

## Features

- Read documents directly from ChromaDB SQLite databases
- Support for multiple collections
- Simple text-based search across collections
- Convert documents to Pandas DataFrames
- Retrieve documents with search query

## Usage

```python
from utils.chroma_parser import ChromaParser

# Initialize the parser with default vector_db_separate directory
parser = ChromaParser()

# Get documents from a collection
docs = parser.get_collection_documents("inventory")

# Search in a collection
results = parser.search_collection("inventory", "pharmaceutical inventory", top_k=5)

# Get a DataFrame of documents
df = parser.get_dataframe("inventory")

# Get all DataFrames with a search query
dfs = parser.get_all_dataframes_with_query("temperature control requirements")
```

## Collections Structure

The parser is configured for the following collections:

- `inventory`: Pharmaceutical inventory data
- `transport`: Transport and shipping data
- `guidelines`: Regulatory guidelines
- `policies`: Company policies

## Database Schema

The parser connects to the ChromaDB SQLite database and retrieves documents from the following tables:

- `embeddings`: Main table with embedding IDs
- `embedding_metadata`: Metadata for embeddings, including document content

## Testing

A test script is available to verify the functionality of the ChromaParser:

```bash
python test_parser.py
```

The test script checks:
1. Document counts in each collection
2. Search functionality with multiple queries
3. DataFrame conversion
4. Retrieval of dataframes with search queries

## Limitations

- This is a direct parser that doesn't use the ChromaDB API
- Search is based on simple text matching, not vector similarity
- No support for adding or updating documents 