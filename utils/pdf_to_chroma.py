import os
import logging
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from config.settings import AppConfig
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chroma_conversion.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

class PdfToChromaConverter:
    """Converts PDF files to ChromaDB collections."""
    
    def __init__(self, config=None):
        """Initialize the ChromaDB converter with configuration."""
        self.config = config or AppConfig()
        load_dotenv()
        
        # Set up the embedding function using the model specified in config
        self.embedding_function = OpenAIEmbeddings(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=self.config.EMBEDDING_MODEL
        )
        
        # Define source PDF files and their corresponding collection names
        self.sources = {
            'inventory': {
                'pdf_path': 'datasets/inventory_data.pdf',
                'collection_name': 'inventory'
            },
            'transport': {
                'pdf_path': 'datasets/transport_history.pdf',
                'collection_name': 'transport'
            },
            'guidelines': {
                'pdf_path': 'datasets/US Government Guidelines for Medicine Transportation and Storage.pdf',
                'collection_name': 'guidelines'
            },
            'policy': {
                'pdf_path': 'datasets/Inventory Management Policy.pdf',
                'collection_name': 'policy'
            }
        }
        
        # Create text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Create chroma_db directory if it doesn't exist
        os.makedirs('chroma_db', exist_ok=True)
        
        logger.info("PdfToChromaConverter initialized")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            logger.info(f"Extracting text from {pdf_path}")
            reader = PdfReader(pdf_path)
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    def process_text_to_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split text into chunks and create LangChain documents."""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        **metadata,
                        "chunk": i,
                        "chunk_count": len(chunks)
                    }
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error processing text into chunks: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def process_inventory_pdf(self, pdf_path: str) -> List[Document]:
        """Process inventory PDF to extract structured data."""
        try:
            logger.info(f"Processing inventory data from {pdf_path}")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # For inventory data, we'll try to parse it into a more structured format
            # Split by lines and look for data rows
            lines = text.split('\n')
            data_lines = []
            
            # Look for header and data rows
            header_found = False
            header = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for headers
                if "ItemID" in line and "GenericName" in line and not header_found:
                    header = line.split()
                    header_found = True
                    continue
                
                # If we found a header, try to parse data rows
                if header_found and any(char.isdigit() for char in line):
                    data_lines.append(line)
            
            # Process into documents
            documents = []
            
            # If we couldn't parse structured data, fall back to chunking
            if not header_found or not data_lines:
                logger.warning("Could not parse structured inventory data, falling back to text chunks")
                return self.process_text_to_chunks(text, {"source": "inventory", "type": "inventory_data"})
            
            # Process into documents with more structured metadata
            for i, line in enumerate(data_lines):
                # Create document with the line as content
                doc = Document(
                    page_content=line,
                    metadata={
                        "source": "inventory",
                        "type": "inventory_data",
                        "line_number": i,
                        "total_lines": len(data_lines)
                    }
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} inventory documents")
            return documents
        except Exception as e:
            logger.error(f"Error processing inventory PDF: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def process_pdf(self, source_name: str) -> bool:
        """Process a PDF file and create a ChromaDB collection."""
        try:
            source_info = self.sources[source_name]
            pdf_path = source_info['pdf_path']
            collection_name = source_info['collection_name']
            
            logger.info(f"Processing {source_name} data from {pdf_path}")
            
            # Check if PDF file exists
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            # Special processing for inventory data
            if source_name == 'inventory':
                documents = self.process_inventory_pdf(pdf_path)
            else:
                # For other PDFs, extract text and chunk it
                text = self.extract_text_from_pdf(pdf_path)
                if not text:
                    logger.error(f"Could not extract text from {pdf_path}")
                    return False
                
                # Process text into chunks
                documents = self.process_text_to_chunks(text, {
                    "source": source_name, 
                    "type": f"{source_name}_data"
                })
            
            if not documents:
                logger.error(f"No documents created for {source_name}")
                return False
            
            # Create ChromaDB collection
            collection_path = os.path.join('chroma_db', collection_name)
            
            # Create the vector store with documents
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                collection_name=collection_name,
                persist_directory=collection_path
            )
            
            # Persist the collection
            vectorstore.persist()
            logger.info(f"Successfully created ChromaDB collection for {source_name} with {len(documents)} documents")
            
            return True
        
        except Exception as e:
            logger.error(f"Error processing {source_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def convert_all(self):
        """Convert all PDF sources to ChromaDB collections."""
        results = {}
        for source_name in self.sources.keys():
            success = self.process_pdf(source_name)
            results[source_name] = "Success" if success else "Failed"
        
        logger.info("Conversion results:")
        for source, result in results.items():
            logger.info(f"  {source}: {result}")
        
        return results

if __name__ == "__main__":
    logger.info("Starting PDF to ChromaDB conversion process")
    converter = PdfToChromaConverter()
    results = converter.convert_all()
    logger.info("PDF to ChromaDB conversion process completed") 