import json
from config.settings import AppConfig
import pandas as pd

class ReportGenerator:
    def __init__(self):
        self.config = AppConfig()
        
    def generate(self, analysis_result: dict) -> dict:
        """
        Generate a final report from the analysis results.
        
        Args:
            analysis_result (dict): The analysis results to format
            
        Returns:
            dict: Formatted report with insights and visualizations
        """
        # Extract components from analysis
        insights = analysis_result.get('insights', '')
        formatted_response = analysis_result.get('formatted_response', '')
        charts = analysis_result.get('charts', [])
        has_numerical_data = analysis_result.get('has_numerical_data', False)
        
        # Format the report
        report = {
            'text': formatted_response if formatted_response else insights,
            'insights': insights,
            'visualizations': charts if has_numerical_data else [],
            'metadata': {
                'has_visualizations': bool(charts and has_numerical_data),
                'visualization_count': len(charts) if has_numerical_data else 0,
                'has_table': '|' in formatted_response if formatted_response else False,
                'generated_at': pd.Timestamp.now().isoformat()
            }
        }
        
        return report 