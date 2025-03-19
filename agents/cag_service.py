import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import pandas as pd
from pypdf import PdfReader
from datetime import datetime
import json
import openai
import torch

# Conditionally import transformers only if needed
transformers_available = False
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.cache_utils import DynamicCache
    transformers_available = True
except ImportError:
    pass

from loguru import logger
import numpy as np

# Load environment variables
load_dotenv()

class CAGService:
    """
    Cache Augmented Generation service for pharmaceutical supply chain documents.
    Implements a system that preloads documents to avoid retrievals during inference.
    """
    
    def __init__(self, model_name: str = None, force_use_openai: bool = True):
        """Initialize the CAG service with OpenAI or local model."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        # Determine if we're using OpenAI or local model
        self.use_openai = True
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # For local models (if available and requested)
        if not force_use_openai and transformers_available and model_name and not model_name.startswith("gpt"):
            try:
                self.use_openai = False
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Configure model for 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                
                self.kv_cache = DynamicCache()
                logger.info(f"Successfully initialized local model: {model_name}")
            except Exception as e:
                logger.error(f"Error initializing local model: {e}")
                logger.info("Falling back to OpenAI")
                self.use_openai = True
        else:
            if not transformers_available:
                logger.info("Transformers library not available, using OpenAI")
            elif force_use_openai:
                logger.info("Forced to use OpenAI as specified")
        
        # Initialize OpenAI client
        if self.use_openai:
            self.client = openai.OpenAI(api_key=self.openai_api_key)
            logger.info(f"Using OpenAI model: {self.model_name}")
        
        # Document storage by type
        self.context_data = {}
        self.transport_data = {}
        self.policy_data = {}
        self.guidelines_data = {}
        self.all_combined_context = ""
        
        logger.info("CAG Service initialized successfully")
    
    def preload_knowledge(self, data_dir: str):
        """Preload knowledge from data directory."""
        try:
            # Process files in the data directory
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                
                # Skip inventory data - handled by RAG
                if "inventory" in file.lower() and file.endswith(".csv"):
                    continue
                
                if file.endswith('.pdf'):
                    self._process_pdf(file_path)
                elif file.endswith('.csv') and "transport" in file.lower():
                    self._process_transport_csv(file_path)
            
            # Build combined context
            self._build_combined_context()
            
            # Update cache if using local model
            if not self.use_openai:
                self._update_cache()
                
            logger.info("Knowledge preloaded successfully")
        except Exception as e:
            logger.error(f"Error preloading knowledge: {e}")
    
    def _process_pdf(self, file_path: str):
        """Process a PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                content = ""
                
                # Extract text from each page
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        content += f"\n--- Page {i+1} ---\n{text}\n"
                
                file_name = os.path.basename(file_path)
                
                # Store content based on document type
                self.context_data[file_path] = {
                    'type': 'pdf',
                    'content': content,
                    'title': file_name,
                    'pages': len(reader.pages),
                    'last_updated': datetime.now()
                }
                
                # Categorize document
                if "policy" in file_name.lower() or "policies" in file_name.lower():
                    self.policy_data[file_path] = self.context_data[file_path]
                elif "guideline" in file_name.lower() or "guidelines" in file_name.lower():
                    self.guidelines_data[file_path] = self.context_data[file_path]
                
                logger.info(f"Processed PDF: {file_name}")
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
    
    def _process_transport_csv(self, file_path: str):
        """Process a transport CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Convert to text format
            text_content = "Transport History Data:\n"
            text_content += df.to_string(index=False)
            
            file_name = os.path.basename(file_path)
            
            # Store as transport data
            self.transport_data[file_path] = {
                'type': 'csv',
                'content': text_content,
                'title': file_name,
                'rows': len(df),
                'columns': df.columns.tolist(),
                'last_updated': datetime.now(),
                'dataframe': df  # Keep reference to DataFrame for analytics
            }
            
            # Add to context data
            self.context_data[file_path] = self.transport_data[file_path]
            
            logger.info(f"Processed Transport CSV: {file_name}")
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
    
    def _build_combined_context(self):
        """Build combined context from all documents."""
        context_parts = []
        
        # Add policy documents first
        for file_path, data in self.policy_data.items():
            context_parts.append(f"--- BEGIN {data['title']} ---\n{data['content']}\n--- END {data['title']} ---\n")
        
        # Add guidelines documents
        for file_path, data in self.guidelines_data.items():
            context_parts.append(f"--- BEGIN {data['title']} ---\n{data['content']}\n--- END {data['title']} ---\n")
        
        # Add transport data
        for file_path, data in self.transport_data.items():
            # Only include a summary of transport data to save tokens
            df = data.get('dataframe')
            if df is not None:
                summary = f"--- BEGIN {data['title']} SUMMARY ---\n"
                summary += f"This dataset contains {len(df)} transport records with columns: {', '.join(df.columns)}\n"
                summary += f"Sample data (first 5 rows):\n{df.head().to_string()}\n"
                summary += f"--- END {data['title']} SUMMARY ---\n"
                context_parts.append(summary)
        
        # Build combined context
        self.all_combined_context = "\n\n".join(context_parts)
    
    def _update_cache(self):
        """Update the KV cache with current context."""
        try:
            if self.use_openai:
                return
                
            # Tokenize content for local model
            inputs = self.tokenizer(self.all_combined_context, return_tensors="pt").to(self.model.device)
            
            # Generate KV cache
            with torch.no_grad():
                outputs = self.model(**inputs, use_cache=True)
                self.kv_cache = outputs.past_key_values
                
            logger.info("KV cache updated successfully")
        except Exception as e:
            logger.error(f"Error updating KV cache: {e}")
    
    def query(self, question: str, context_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the cached context.
        
        Args:
            question: The query to answer
            context_type: Type of context to use (policy, guidelines, transport, or None for all)
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Get relevant context based on type
            relevant_context = self._get_relevant_context(context_type)
            
            # Build prompt
            prompt = self._build_prompt(question, relevant_context)
            
            # Generate response
            if self.use_openai:
                response_text = self._query_openai(prompt)
            else:
                response_text = self._query_local_model(prompt)
            
            return {
                'response': response_text,
                'source_type': 'cag',
                'context_type': context_type or 'all',
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'context_length': len(relevant_context)
                }
            }
        except Exception as e:
            logger.error(f"Error during CAG query: {e}")
            return {
                'response': f"I apologize, but I encountered an error while processing your request: {str(e)}",
                'source_type': 'cag',
                'error': str(e)
            }
    
    def _get_relevant_context(self, context_type: Optional[str]) -> str:
        """Get relevant context based on type."""
        if not context_type:
            return self.all_combined_context
            
        context_parts = []
        
        if context_type == 'policy':
            for data in self.policy_data.values():
                context_parts.append(f"--- BEGIN {data['title']} ---\n{data['content']}\n--- END {data['title']} ---\n")
        elif context_type == 'guidelines':
            for data in self.guidelines_data.values():
                context_parts.append(f"--- BEGIN {data['title']} ---\n{data['content']}\n--- END {data['title']} ---\n")
        elif context_type == 'transport':
            for data in self.transport_data.values():
                if 'dataframe' in data:
                    df = data['dataframe']
                    content = f"--- BEGIN {data['title']} ---\n"
                    content += df.to_string(index=False)
                    content += f"\n--- END {data['title']} ---\n"
                    context_parts.append(content)
        
        return "\n\n".join(context_parts) if context_parts else self.all_combined_context
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt with context and question."""
        return f"""You are an expert in pharmaceutical supply chain management with access to the following information:

{context}

Based on this information, please answer the following question thoroughly and accurately:

{question}

If the information to answer the question is not present in the context, please indicate that clearly.
"""
    
    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI with prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical supply chain expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            raise
    
    def _query_local_model(self, prompt: str) -> str:
        """Query local model with prompt and KV cache."""
        try:
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response using KV cache
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    past_key_values=self.kv_cache,
                    max_new_tokens=500,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split(prompt)[-1].strip()
        except Exception as e:
            logger.error(f"Error querying local model: {e}")
            raise
    
    def reset_cache(self):
        """Reset the KV cache."""
        if not self.use_openai:
            self.kv_cache = DynamicCache()
            self._update_cache()
            logger.info("KV cache reset successfully") 