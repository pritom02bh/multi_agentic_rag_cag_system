from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import AppConfig
import os
import re
import logging
import traceback
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)

class QueryEnhancer:
    """Enhances user queries for better retrieval and analysis."""
    
    def __init__(self, config=None):
        """Initialize the QueryEnhancer with configuration.
        
        Args:
            config: Application configuration instance
        """
        self.config = config or AppConfig()
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)
            logger.info("QueryEnhancer initialized with OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
        
        # Define system and enhancement prompts
        self.system_prompt = """You are an expert query enhancer for a pharmaceutical supply chain system.
Your role is to analyze user queries and enhance them for better semantic search and response generation.

Consider these pharmaceutical inventory and transport aspects when enhancing:
1. Inventory management (stock levels, reorder points, expiry dates)
2. Supply chain logistics (shipment status, delays, transport conditions)
3. Regulatory requirements (storage conditions, handling procedures)
4. Business metrics (inventory value, stockout costs, carrying costs)

Enhance the query by:
1. Adding relevant pharmaceutical terminology and expanding acronyms (PHMSA, FDA, TSA, UN 3373)
2. Including appropriate metrics and contextual elements
3. Clarifying ambiguous terms such as "critical stock" or "high demand" with specific thresholds
4. Maintaining the original query's intent
5. Ensuring it's optimized for semantic search
6. Expanding drug names with common synonyms or generic/brand name equivalents

Your enhanced query should be concise and clear."""

        # Initialize pharmaceutical entity dictionary
        self._init_pharma_dictionary()
        
        # Query intent classification patterns
        self.intent_patterns = {
            'factual': [
                r'what is', r'does .+ require', r'is .+ required', 
                r'what (packaging|temperature|regulations)', r'are special',
                r'what are the (labeling|quarantine|compliance)'
            ],
            'analytical': [
                r'how many', r'compare', r'trend', r'which port', r'breakdown',
                r'how much', r'list all', r'average', r'ratio', r'turnover'
            ],
            'hybrid': [
                r'missing .+ logs', r'(documentation|compliance) issues',
                r'had .+ issues', r'violations'
            ]
        }
        
        # Query template
        self.query_template = """
Original Query: {query}

Enhanced Query:"""
        
        # Create prompt template for LangChain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.query_template)
        ])
    
    def _init_pharma_dictionary(self):
        """Initialize pharmaceutical terminology dictionaries."""
        # Medication synonyms and classifications
        self.medication_dict = {
            'paracetamol': ['acetaminophen', 'tylenol', 'panadol', 'analgesic', 'antipyretic'],
            'aspirin': ['acetylsalicylic acid', 'asa', 'nsaid', 'anti-inflammatory'],
            'ibuprofen': ['advil', 'motrin', 'nsaid', 'anti-inflammatory'],
            'insulin': ['humalog', 'novolog', 'lantus', 'hormone', 'antidiabetic', 'biologic'],
            'metformin': ['glucophage', 'antidiabetic', 'oral hypoglycemic'],
            'lisinopril': ['prinivil', 'zestril', 'ace inhibitor', 'antihypertensive'],
            'atorvastatin': ['lipitor', 'statin', 'antihyperlipidemic'],
            'amoxicillin': ['amoxil', 'antibiotic', 'penicillin'],
            'ciprofloxacin': ['cipro', 'antibiotic', 'fluoroquinolone'],
            'albuterol': ['proventil', 'ventolin', 'bronchodilator', 'anti-asthmatic'],
            'fluticasone': ['flonase', 'flovent', 'corticosteroid', 'anti-inflammatory'],
            'adalimumab': ['humira', 'biologic', 'tnf inhibitor', 'immunosuppressant'],
            'rituximab': ['rituxan', 'biologic', 'monoclonal antibody'],
            'erythropoietin': ['epo', 'epogen', 'procrit', 'biologic', 'hormone'],
            'sildenafil': ['viagra', 'revatio', 'pde5 inhibitor'],
            'flu vaccine': ['influenza vaccine', 'flu shot', 'vaccine', 'biologic'],
            'alpha interferon': ['interferon alfa', 'intron a', 'biologic', 'immunomodulator'],
            'beta interferon': ['interferon beta', 'avonex', 'rebif', 'biologic', 'immunomodulator'],
            'growth hormone': ['somatotropin', 'humatrope', 'norditropin', 'biologic', 'hormone'],
            'hydrocortisone': ['cortisol', 'corticosteroid', 'anti-inflammatory']
        }
        
        # Regulatory acronyms and terminology
        self.regulatory_dict = {
            'phmsa': ['pipeline and hazardous materials safety administration', 'hazardous materials', 'dot'],
            'fda': ['food and drug administration', 'regulatory', 'compliance'],
            'tsa': ['transportation security administration', 'security', 'screening'],
            'un 3373': ['biological substance category b', 'diagnostic specimen', 'clinical specimen'],
            'triple packaging': ['primary receptacle', 'secondary packaging', 'outer packaging'],
            'excursion': ['temperature excursion', 'deviation', 'out-of-spec', 'oos'],
            'temperature log': ['temperature monitoring', 'temperature tracking', 'cold chain verification']
        }
        
        # Classification terminology
        self.classification_dict = {
            'class a': ['high value', 'critical', 'essential', 'tier 1'],
            'class b': ['medium value', 'important', 'tier 2'],
            'class c': ['low value', 'routine', 'tier 3'],
            'critical stock': ['essential medication', 'vital inventory', 'life-saving drug'],
            'high demand': ['fast-moving', 'high-turnover', 'high-volume'],
            'stockout': ['zero inventory', 'out-of-stock', 'supply shortage']
        }
    
    def detect_entities(self, query: str) -> Dict[str, List[str]]:
        """Detect pharmaceutical entities in the query."""
        query_lower = query.lower()
        detected = {
            'medications': [],
            'regulatory': [],
            'classification': []
        }
        
        # Detect medications
        for med, synonyms in self.medication_dict.items():
            if med in query_lower or any(syn in query_lower for syn in synonyms):
                detected['medications'].append(med)
        
        # Detect regulatory terms
        for term, synonyms in self.regulatory_dict.items():
            if term in query_lower or any(syn in query_lower for syn in synonyms):
                detected['regulatory'].append(term)
        
        # Detect classification terms
        for term, synonyms in self.classification_dict.items():
            if term in query_lower or any(syn in query_lower for syn in synonyms):
                detected['classification'].append(term)
        
        return detected
    
    def classify_query_intent(self, query: str) -> str:
        """Classify query as factual, analytical, or hybrid."""
        query_lower = query.lower()
        
        # Check for hybrid patterns first (most specific)
        for pattern in self.intent_patterns['hybrid']:
            if re.search(pattern, query_lower):
                return 'hybrid'
        
        # Check for analytical patterns
        for pattern in self.intent_patterns['analytical']:
            if re.search(pattern, query_lower):
                return 'analytical'
        
        # Default to factual
        return 'factual'
    
    def expand_query_with_synonyms(self, query: str, entities: Dict[str, List[str]]) -> str:
        """Expand query with relevant synonyms based on detected entities."""
        expanded_query = query
        
        # Only add up to 3 most relevant synonyms to keep query concise
        for med in entities['medications']:
            synonyms = self.medication_dict.get(med, [])[:2]  # Limit to 2 synonyms
            if synonyms:
                expanded_terms = " " + " ".join(synonyms)
                expanded_query += f" ({med}:{expanded_terms})"
        
        # Add regulatory term expansions
        for term in entities['regulatory']:
            synonyms = self.regulatory_dict.get(term, [])[:1]  # Limit to 1 synonym
            if synonyms:
                expanded_query += f" ({term}:{synonyms[0]})"
        
        return expanded_query
    
    def validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        """Validate the query for common issues."""
        try:
            if not query or not query.strip():
                return False, "Query cannot be empty"
                
            if len(query) < 3:
                return False, "Query is too short"
                
            if len(query) > 1000:
                return False, "Query is too long (max 1000 characters)"
                
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating query: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        try:
            # Remove extra whitespace
            query = ' '.join(query.split())
            
            # Remove special characters except basic punctuation
            query = re.sub(r'[^\w\s.,?!-]', '', query)
            
            # Ensure query ends with proper punctuation
            if not query.endswith(('?', '!', '.')):
                query += '?'
                
            return query
            
        except Exception as e:
            logger.error(f"Error cleaning query: {str(e)}")
            return query  # Return original query if cleaning fails
    
    def enhance(self, query: str) -> Dict[str, Any]:
        """Enhance the user query to make it more effective for searching."""
        logger.info(f"Starting query enhancement for: '{query}'")
        
        try:
            # Validate and clean query
            is_valid, error_message = self.validate_query(query)
            if not is_valid:
                logger.error(f"Query validation failed: {error_message}")
                return {
                    'status': 'error',
                    'message': error_message,
                    'enhanced_query': None,
                    'original_query': query
                }
            
            cleaned_query = self.clean_query(query)
            logger.info(f"Cleaned query: '{cleaned_query}'")
            
            # Detect entities and classify intent
            entities = self.detect_entities(cleaned_query)
            query_intent = self.classify_query_intent(cleaned_query)
            
            logger.info(f"Detected entities: {entities}")
            logger.info(f"Query intent classification: {query_intent}")
            
            # Check for web search patterns
            is_web_search_query = self._check_web_search_patterns(cleaned_query)
            
            if is_web_search_query:
                logger.info(f"Detected query that likely needs web search: '{cleaned_query}'")
                return {
                    'status': 'success',
                    'message': 'Query preserved for web search',
                    'original_query': query,
                    'cleaned_query': cleaned_query,
                    'enhanced_query': cleaned_query,  # Keep the original intent for web search
                    'is_web_search': True,
                    'entities': entities,
                    'query_intent': query_intent
                }
            
            # For hybrid queries, optionally expand with synonyms
            expanded_query = cleaned_query
            if query_intent == 'hybrid':
                expanded_query = self.expand_query_with_synonyms(cleaned_query, entities)
                logger.info(f"Expanded query with synonyms: '{expanded_query}'")
            
            # Enhance query using OpenAI
            try:
                response = self.client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.query_template.format(query=expanded_query)}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                
                enhanced_query = response.choices[0].message.content.strip()
                logger.info(f"Enhanced query: '{enhanced_query}'")
                
                if not enhanced_query:
                    return {
                        'status': 'error',
                        'message': 'Query enhancement returned empty result',
                        'enhanced_query': expanded_query,  # Fall back to expanded query
                        'original_query': query,
                        'entities': entities,
                        'query_intent': query_intent
                    }
                
                return {
                    'status': 'success',
                    'message': 'Query enhanced successfully',
                    'original_query': query,
                    'cleaned_query': cleaned_query,
                    'enhanced_query': enhanced_query,
                    'is_web_search': False,
                    'entities': entities,
                    'query_intent': query_intent
                }
                
            except Exception as e:
                logger.error(f"OpenAI query enhancement failed: {str(e)}")
                logger.error(traceback.format_exc())
                # Fall back to expanded query
                return {
                    'status': 'error',
                    'message': f"OpenAI error: {str(e)}",
                    'enhanced_query': expanded_query,
                    'original_query': query,
                    'entities': entities,
                    'query_intent': query_intent
                }
            
        except Exception as e:
            logger.error(f"Unexpected error in query enhancement: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f"Enhancement error: {str(e)}",
                'enhanced_query': query,
                'original_query': query
            }
    
    def _check_web_search_patterns(self, query: str) -> bool:
        """Check if query matches web search patterns."""
        web_search_patterns = [
            r'weather\s+in|temperature\s+in|temperature\s+of|what\s+is\s+the\s+temperature|how\s+hot\s+is|how\s+cold\s+is', # Weather patterns
            r'news\s+about|latest\s+news|recent\s+news|current\s+events', # News patterns
            r'stock\s+price\s+of|market\s+value\s+of|how\s+is\s+the\s+stock\s+market', # Stock market patterns
            r'sports\s+scores|game\s+results|who\s+won\s+the\s+game', # Sports patterns
            r'current\s+time\s+in|what\s+time\s+is\s+it|what\s+is\s+the\s+time', # Time queries
            r'population\s+of|how\s+many\s+people\s+live\s+in', # Demographics
            r'currency\s+exchange|exchange\s+rate|conversion\s+rate', # Currency
            r'movie\s+times|show\s+times|theater\s+schedule', # Entertainment
            r'traffic\s+in|traffic\s+conditions|traffic\s+report', # Traffic
            r'restaurant\s+in|places\s+to\s+eat|cafes\s+in', # Places
            r'tariff\s+on|tariffs\s+for|import\s+duty|export\s+duty|customs\s+duty', # Tariffs
            r'tax\s+rate|sales\s+tax|vat|value\s+added\s+tax|import\s+tax', # Taxes
            r'shipping\s+cost|transport\s+fees|logistics\s+charges', # Shipping costs
            r'regulatory\s+changes|regulation\s+updates|new\s+laws|legislation', # Regulatory updates
            r'market\s+prices|commodity\s+prices|raw\s+material\s+costs', # Market prices
            r'international\s+trade|trade\s+agreements|trade\s+policies', # Trade info
            r'economic\s+indicators|gdp|inflation\s+rate|unemployment', # Economic indicators
            r'global\s+events|political\s+situation|geopolitical', # Global events
            r'exchange\s+rates|currency\s+value|foreign\s+exchange', # Forex
            r'supply\s+chain\s+disruptions|logistics\s+issues|shipping\s+delays', # Supply chain issues
        ]
        
        return any(re.search(pattern, query.lower()) for pattern in web_search_patterns) 