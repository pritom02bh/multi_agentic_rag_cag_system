from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import AppConfig
import os

class Router:
    def __init__(self):
        self.config = AppConfig()
        self.llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=self.config.OPENAI_MODEL,
            temperature=0.1  # Low temperature for consistent routing
        )
        self.prompt = ChatPromptTemplate.from_template(
            self.config.ROUTER_PROMPT
        )
        
    def route(self, query: str) -> str:
        """
        Route the query to the most appropriate data source.
        
        Args:
            query (str): The enhanced query
            
        Returns:
            str: Selected data source
        """
        chain = self.prompt | self.llm
        response = chain.invoke({"query": query})
        
        # Extract and validate the source
        selected_source = response.content.strip().lower()
        
        if selected_source not in self.config.DATA_SOURCES:
            # Default to inventory if invalid source
            selected_source = 'inventory'
            
        return selected_source 