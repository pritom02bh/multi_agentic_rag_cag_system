from typing import Any, Dict, List, Optional, Union
from langchain_community.tools import BaseTool
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import pandas as pd
import numpy as np
from datetime import datetime
import json
import aiohttp
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import base64
from io import BytesIO
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
import logging

# Load environment variables
load_dotenv()

@dataclass
class AgentOutput:
    agent_type: str
    content: Any
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class OutputProcessor:
    def __init__(self):
        self.outputs: Dict[str, List[AgentOutput]] = {}
        self.visualizations: List[Dict[str, Any]] = []
        self.combined_response: Dict[str, Any] = {
            "text_response": "",
            "visualizations": [],
            "sources": [],
            "confidence": 0.0,
            "metadata": {}
        }
    
    def add_output(self, agent_output: AgentOutput) -> None:
        """Add an agent's output to the collection"""
        if agent_output.agent_type not in self.outputs:
            self.outputs[agent_output.agent_type] = []
        self.outputs[agent_output.agent_type].append(agent_output)
    
    def add_visualization(self, viz_data: Dict[str, Any]) -> None:
        """Add a visualization to the collection"""
        self.visualizations.append(viz_data)
    
    def process_outputs(self) -> Dict[str, Any]:
        """Process all collected outputs and return a combined response"""
        # Process text responses
        text_responses = []
        sources = []
        total_confidence = 0
        num_responses = 0
        
        for agent_type, outputs in self.outputs.items():
            for output in outputs:
                if isinstance(output.content, str):
                    text_responses.append(output.content)
                elif isinstance(output.content, dict):
                    if "text" in output.content:
                        text_responses.append(output.content["text"])
                    if "sources" in output.content:
                        sources.extend(output.content["sources"])
                
                total_confidence += output.confidence
                num_responses += 1
                
                # Add metadata
                self.combined_response["metadata"][agent_type] = {
                    "timestamp": output.timestamp.isoformat(),
                    "confidence": output.confidence,
                    **output.metadata
                }
        
        # Combine text responses
        self.combined_response["text_response"] = "\n\n".join(text_responses)
        self.combined_response["sources"] = list(set(sources))  # Remove duplicates
        self.combined_response["confidence"] = total_confidence / num_responses if num_responses > 0 else 0
        self.combined_response["visualizations"] = self.visualizations
        
        return self.combined_response
    
    def get_agent_outputs(self, agent_type: str) -> List[AgentOutput]:
        """Get all outputs from a specific agent"""
        return self.outputs.get(agent_type, [])
    
    def clear(self) -> None:
        """Clear all collected outputs"""
        self.outputs.clear()
        self.visualizations.clear()
        self.combined_response = {
            "text_response": "",
            "visualizations": [],
            "sources": [],
            "confidence": 0.0,
            "metadata": {}
        }

# Create a global instance of OutputProcessor
output_processor = OutputProcessor()

class CAGTools:
    @staticmethod
    def cache_lookup() -> Tool:
        def _cache_lookup(query_hash: str) -> Optional[str]:
            # TODO: Implement actual cache lookup using Redis
            return None
        
        return Tool(
            name="cache_lookup",
            func=_cache_lookup,
            description="Look up cached responses for similar queries"
        )
    
    @staticmethod
    def cache_store() -> Tool:
        def _cache_store(query_hash: str, response: str) -> bool:
            # TODO: Implement actual cache storage using Redis
            return True
        
        return Tool(
            name="cache_store",
            func=_cache_store,
            description="Store responses in cache for future use"
        )

class RAGTools:
    @staticmethod
    def vector_search() -> Tool:
        def _vector_search(query: str, k: int = 5, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
            try:
                # Initialize Pinecone
                pinecone.init(
                    api_key=os.getenv('PINECONE_API_KEY'),
                    environment=os.getenv('PINECONE_ENV')
                )
                index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME'))
                
                # Initialize embeddings
                embeddings = OpenAIEmbeddings(
                    api_key=os.getenv('OPENAI_API_KEY')
                )
                
                # Get query embedding
                query_embedding = embeddings.embed_query(query)
                
                # Perform vector search
                search_response = index.query(
                    vector=query_embedding,
                    top_k=k,
                    namespace="pharmaceutical_supply_chain",
                    filter=metadata_filter,
                    include_metadata=True
                )
                
                # Process results
                results = []
                for match in search_response.matches:
                    results.append({
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata,
                        "text": match.metadata.get("text", "")
                    })
                
                # Add to output processor
                output = AgentOutput(
                    agent_type="rag_vector_search",
                    content={
                        "query": query,
                        "results": results
                    },
                    confidence=max([r["score"] for r in results]) if results else 0.0,
                    metadata={
                        "num_results": len(results),
                        "metadata_filter": metadata_filter
                    }
                )
                output_processor.add_output(output)
                
                return results
                
            except Exception as e:
                logger.error(f"Error in vector search: {str(e)}")
                return []
        
        return Tool(
            name="vector_search",
            func=_vector_search,
            description="Search vector store for relevant documents with optional metadata filtering"
        )
    
    @staticmethod
    def document_retrieval() -> Tool:
        def _document_retrieval(doc_ids: Union[str, List[str]], metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
            try:
                # Initialize Pinecone
                pinecone.init(
                    api_key=os.getenv('PINECONE_API_KEY'),
                    environment=os.getenv('PINECONE_ENV')
                )
                index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME'))
                
                # Convert single ID to list
                if isinstance(doc_ids, str):
                    doc_ids = [doc_ids]
                
                # Fetch documents
                fetch_response = index.fetch(
                    ids=doc_ids,
                    namespace="pharmaceutical_supply_chain"
                )
                
                # Process results
                results = []
                for doc_id, vector_data in fetch_response.vectors.items():
                    if metadata_filter:
                        # Apply metadata filter
                        if not all(vector_data.metadata.get(k) == v for k, v in metadata_filter.items()):
                            continue
                    
                    results.append({
                        "id": doc_id,
                        "metadata": vector_data.metadata,
                        "text": vector_data.metadata.get("text", "")
                    })
                
                # Add to output processor
                output = AgentOutput(
                    agent_type="rag_document_retrieval",
                    content={
                        "doc_ids": doc_ids,
                        "results": results
                    },
                    confidence=1.0 if results else 0.0,
                    metadata={
                        "num_results": len(results),
                        "metadata_filter": metadata_filter
                    }
                )
                output_processor.add_output(output)
                
                return results
                
            except Exception as e:
                logger.error(f"Error in document retrieval: {str(e)}")
                return []
        
        return Tool(
            name="document_retrieval",
            func=_document_retrieval,
            description="Retrieve full documents by ID with optional metadata filtering"
        )

class WebSearchTools:
    @staticmethod
    def web_search() -> Tool:
        search_wrapper = DuckDuckGoSearchAPIWrapper()
        search_tool = DuckDuckGoSearchRun(api_wrapper=search_wrapper)
        
        def _process_search(query: str) -> Dict[str, Any]:
            result = search_tool.run(query)
            output = AgentOutput(
                agent_type="web_search",
                content={"text": result, "sources": []},
                confidence=0.8,
                metadata={"query": query}
            )
            output_processor.add_output(output)
            return result
        
        return Tool(
            name="web_search",
            func=_process_search,
            description="Perform web search for current information using DuckDuckGo"
        )
    
    @staticmethod
    def news_search() -> Tool:
        def _news_search(query: str) -> Dict[str, Any]:
            import requests
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": os.getenv('NEWS_API_KEY'),
                "sortBy": "relevancy",
                "language": "en"
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                articles = [
                    {
                        "title": article["title"],
                        "description": article["description"],
                        "url": article["url"],
                        "publishedAt": article["publishedAt"],
                        "source": article["source"]["name"]
                    }
                    for article in data.get("articles", [])[:5]
                ]
                
                output = AgentOutput(
                    agent_type="news_search",
                    content={
                        "text": "\n".join([f"{a['title']}: {a['description']}" for a in articles]),
                        "sources": [a["url"] for a in articles]
                    },
                    confidence=0.9,
                    metadata={"query": query, "num_results": len(articles)}
                )
                output_processor.add_output(output)
                return articles
            return []
        
        return Tool(
            name="news_search",
            func=_news_search,
            description="Search for recent news articles using NewsAPI"
        )
    
    @staticmethod
    def content_extraction() -> Tool:
        async def _extract_content(url: str) -> Dict[str, Any]:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract main content
                    content = soup.get_text()
                    
                    return {
                        "title": soup.title.string if soup.title else "",
                        "content": content,
                        "url": url
                    }
        
        return Tool(
            name="content_extraction",
            func=_extract_content,
            description="Extract and clean content from web pages"
        )

class EnhancedAnalyticsTools:
    @staticmethod
    def data_analysis() -> Tool:
        def _analyze_data(data: Union[str, Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    return {"error": "Invalid data format"}
            
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                return {"error": "Unsupported data format"}
            
            analysis_result = {
                "summary_stats": df.describe().to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "correlations": df.corr().to_dict() if df.select_dtypes(include=[np.number]).columns.size > 1 else {}
            }
            
            output = AgentOutput(
                agent_type="data_analysis",
                content=analysis_result,
                confidence=0.95,
                metadata={"columns": list(df.columns), "rows": len(df)}
            )
            output_processor.add_output(output)
            return analysis_result
        
        return Tool(
            name="data_analysis",
            func=_analyze_data,
            description="Analyze data and provide statistical insights"
        )
    
    @staticmethod
    def visualization() -> Tool:
        def _create_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                df = pd.DataFrame(data)
                
                # Create visualizations based on data types
                viz_data = {}
                
                # Time series if datetime column exists
                date_cols = df.select_dtypes(include=['datetime64']).columns
                if len(date_cols) > 0:
                    date_col = date_cols[0]
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        fig = px.line(df, x=date_col, y=col, title=f"{col} over time")
                        viz_data[f"{col}_timeseries"] = fig.to_json()
                
                # Correlation heatmap for numeric columns
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    fig = px.imshow(numeric_df.corr(), title="Correlation Heatmap")
                    viz_data["correlation_heatmap"] = fig.to_json()
                
                output = AgentOutput(
                    agent_type="visualization",
                    content=viz_data,
                    confidence=0.9,
                    metadata={"viz_types": list(viz_data.keys())}
                )
                output_processor.add_output(output)
                return viz_data
            
            except Exception as e:
                return {"error": str(e)}
        
        return Tool(
            name="visualization",
            func=_create_visualization,
            description="Create visualizations from data"
        )
    
    @staticmethod
    def trend_detection() -> Tool:
        def _detect_trends(data: List[Dict[str, Any]]) -> Dict[str, Any]:
            df = pd.DataFrame(data)
            
            # Perform trend analysis
            trends = []
            for column in df.select_dtypes(include=[np.number]).columns:
                series = df[column]
                
                # Calculate trend statistics
                slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(series)), series)
                
                # Detect seasonality using autocorrelation
                acf = np.correlate(series - series.mean(), series - series.mean(), mode='full') / len(series)
                acf = acf[len(acf)//2:]
                
                trends.append({
                    "feature": column,
                    "trend": {
                        "direction": "increasing" if slope > 0 else "decreasing",
                        "slope": slope,
                        "r_squared": r_value**2,
                        "p_value": p_value,
                        "confidence": 1 - p_value
                    },
                    "seasonality": {
                        "detected": bool(np.any(acf[1:] > 2/np.sqrt(len(series)))),
                        "strength": float(np.max(acf[1:]))
                    }
                })
            
            return {
                "trends": trends,
                "overall_confidence": np.mean([t["trend"]["confidence"] for t in trends])
            }
        
        return Tool(
            name="trend_detection",
            func=_detect_trends,
            description="Detect patterns and trends in time series data"
        )
    
    @staticmethod
    def statistical_analysis() -> Tool:
        def _statistical_analysis(data: List[float], test_type: str) -> Dict[str, Any]:
            series = pd.Series(data)
            
            if test_type == "normality":
                stat, p_value = stats.normaltest(series)
                return {
                    "test_type": "normality",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > 0.05
                }
            elif test_type == "outliers":
                z_scores = np.abs(stats.zscore(series))
                outliers = series[z_scores > 3]
                return {
                    "test_type": "outliers",
                    "outliers": outliers.tolist(),
                    "outlier_indices": outliers.index.tolist(),
                    "z_scores": z_scores.tolist()
                }
            
            return {
                "test_type": test_type,
                "error": "Unsupported test type"
            }
        
        return Tool(
            name="statistical_analysis",
            func=_statistical_analysis,
            description="Perform statistical tests and analysis"
        )

def get_augmented_tools(agent_type: str) -> List[Tool]:
    """Get tools for augmented agents"""
    tools_map = {
        "cag": [
            CAGTools.cache_lookup(),
            CAGTools.cache_store()
        ],
        "rag": [
            RAGTools.vector_search(),
            RAGTools.document_retrieval()
        ],
        "web_search": [
            WebSearchTools.web_search(),
            WebSearchTools.news_search()
        ],
        "enhanced_analytics": [
            EnhancedAnalyticsTools.data_analysis(),
            EnhancedAnalyticsTools.visualization()
        ]
    }
    
    return tools_map.get(agent_type, []) 