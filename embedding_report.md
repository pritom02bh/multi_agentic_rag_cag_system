# Vector Embedding Report for RAG Development

## 1. Embedding Model Details
- **Model**: OpenAI text-embedding-3-small
- **Dimensions**: 1536 dimensions per embedding
- **Encoding Format**: float
- **Context Window**: Up to 8191 tokens
- **Model Properties**:
  - Designed for semantic similarity
  - Supports multilingual content
  - Optimized for RAG applications

## 2. Data Sources and Processing

### 2.1 CSV Files
1. **inventory_data.csv**
   - Type: Structured data
   - Processing: Row-by-row embedding
   - Format: `column_name: value | column_name: value`
   - Metadata preserved: All original columns
   - Location: `datasets/inventory_data.csv`

2. **transport_history.csv**
   - Type: Structured data
   - Processing: Row-by-row embedding
   - Format: `column_name: value | column_name: value`
   - Metadata preserved: All original columns
   - Location: `datasets/transport_history.csv`

### 2.2 PDF Files
1. **US Government Guidelines for Medicine Transportation and Storage.pdf**
   - Type: Unstructured text
   - Processing: Page-by-page embedding
   - Metadata preserved: Page numbers
   - Location: `datasets/US Government Guidelines for Medicine Transportation and Storage.pdf`

2. **Inventory Management Policy.pdf**
   - Type: Unstructured text
   - Processing: Page-by-page embedding
   - Metadata preserved: Page numbers
   - Location: `datasets/Inventory Management Policy.pdf`

## 3. Embedding Storage Format

### 3.1 Output Files (in embedded_data/)
- `inventory_data_embedded.csv`
- `transport_history_embedded.csv`
- `us_government_guidelines_for_medicine_transportation_and_storage_embedded.csv`
- `inventory_management_policy_embedded.csv`

### 3.2 CSV Structure
Each embedded file contains the following columns:
```
- id: Unique identifier for each record
- content: Original text content
- embedding: Comma-separated 1536-dimensional vector
- source: Original file path
- type: Document type (csv/pdf)
- row_index: (CSV only) Original row number
- page_number: (PDF only) Page number
- [original_columns]: (CSV only) All original columns preserved
```

## 4. Text Processing Details

### 4.1 Text Cleaning
```python
- Whitespace normalization
- Special character removal (preserving .,!?-)
- Empty content filtering
```

### 4.2 Text Formatting
- CSV: `column_name: value | column_name: value`
- PDF: Raw page text with preserved formatting

## 5. Using the Embeddings for RAG

### 5.1 Loading Embeddings
```python
import pandas as pd
import numpy as np

def load_embeddings(file_path):
    df = pd.read_csv(file_path)
    # Convert string embeddings back to numpy arrays
    df['embedding_array'] = df['embedding'].apply(
        lambda x: np.array([float(i) for i in x.split(',')])
    )
    return df
```

### 5.2 Similarity Search
```python
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine

def find_similar_documents(query, df, top_k=5):
    # Create embedding for query
    client = OpenAI()
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
        encoding_format="float"
    ).data[0].embedding

    # Calculate similarities
    similarities = []
    for idx, row in df.iterrows():
        similarity = 1 - cosine(query_embedding, row['embedding_array'])
        similarities.append((similarity, idx))
    
    # Sort by similarity
    similarities.sort(reverse=True)
    
    # Return top_k most similar documents
    return df.iloc[[idx for _, idx in similarities[:top_k]]]
```

### 5.3 Hybrid Search Example
```python
def hybrid_search(query, df, metadata_filters=None):
    # First filter by metadata if specified
    if metadata_filters:
        for key, value in metadata_filters.items():
            df = df[df[key] == value]
    
    # Then perform similarity search
    return find_similar_documents(query, df)

# Example usage:
filters = {
    'type': 'pdf',
    'page_number': 1  # First pages only
}
results = hybrid_search("medicine storage temperature", df, filters)
```

## 6. Important Considerations

### 6.1 Performance
- Embedding dimensions: 1536 (fixed)
- Average tokens per document:
  - CSV rows: Varies by row content
  - PDF pages: Full page content

### 6.2 Metadata Usage
- CSV files preserve all original columns for filtering
- PDF files include page numbers for document navigation
- Source tracking for result attribution

### 6.3 Best Practices
1. Use metadata filters before similarity search for efficiency
2. Consider chunking large PDF pages for more granular retrieval
3. Normalize similarity scores based on document types
4. Cache embeddings for frequently used queries

## 7. Limitations and Recommendations

### 7.1 Limitations
- PDF processing is page-based (no cross-page context)
- CSV embeddings include all columns (might include irrelevant data)
- No document chunking implemented (might affect context window)

### 7.2 Recommendations for RAG
1. Implement chunking for PDF documents
2. Add semantic sectioning for better context
3. Consider implementing a caching layer
4. Add relevance scoring based on document type
5. Implement error handling for malformed embeddings 