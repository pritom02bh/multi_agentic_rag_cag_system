from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import AppConfig
import os

class QueryEnhancer:
    def __init__(self):
        self.config = AppConfig()
        self.llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=self.config.OPENAI_MODEL,
            temperature=0.3  # Lower temperature for more focused enhancement
        )
        self.prompt = ChatPromptTemplate.from_template(
            self.config.QUERY_ENHANCEMENT_PROMPT
        )
        
    def enhance(self, query: str) -> str:
        """
        Enhance the user query to make it more effective for searching.
        
        Args:
            query (str): The original user query
            
        Returns:
            str: The enhanced query
        """
        chain = self.prompt | self.llm
        response = chain.invoke({"query": query})
        
        return response.content 