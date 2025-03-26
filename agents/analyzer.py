import re
import logging
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from config.settings import AppConfig

logger = logging.getLogger(__name__)

class AnalyticalAgent:
    """Agent responsible for analyzing inventory and transport data"""
    
    def __init__(self):
        self.config = AppConfig()
    
    def analyze(self, text: str) -> dict:
        """
        Analyze the input text and generate insights
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Analysis results including formatted response and visualizations
        """
        try:
            # Format the data in a structured way
            formatted_response = self._format_tabular_data(text)
            
            # Generate visualizations
            charts = self._generate_charts(text)
            
            return {
                'formatted_response': formatted_response,
                'charts': charts
            }
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {
                'formatted_response': text,
                'charts': []
            }
    
    def _format_tabular_data(self, text: str) -> str:
        """Format data in a tabular structure with enhanced presentation"""
        try:
            # Extract item data using regex
            items = []
            
            # Enhanced patterns for more detailed information
            patterns = {
                'ItemID': r'ITEM-[A-Z0-9]+',
                'GenericName': r'([A-Za-z\s]+)(?=\s*[:|]\s*CurrentStock)',
                'CurrentStock': r'CurrentStock\s*[-:]\s*([\d,]+)',
                'MaxInventory': r'MaxInventory\s*[:]\s*([\d,]+)',
                'ReorderPoint': r'ReorderPoint\s*[:]\s*([\d,.]+)',
                'Status': r'Status\s*[:]\s*([^,\n]+)',
                'LeadTime': r'LeadTime(?:Days)?\s*[:]\s*(\d+)',
                'UnitCost': r'UnitCost\s*[:]\s*\$?([\d,.]+)',
                'SellingPrice': r'SellingPrice\s*[:]\s*\$?([\d,.]+)',
                'LastUpdated': r'LastUpdated\s*[:]\s*([^,\n]+)',
                'BatchNumber': r'BatchNumber\s*[:]\s*([^,\n]+)'
            }
            
            # Find all items and their details
            item_blocks = text.split('\n')
            current_item = {}
            
            for line in item_blocks:
                if not line.strip():
                    if current_item:
                        items.append(current_item)
                        current_item = {}
                    continue
                
                # Extract item details
                for key, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        current_item[key] = match.group(1).strip()
            
            # Add last item if exists
            if current_item:
                items.append(current_item)
            
            # If we found structured data, format it as a professional table
            if items:
                # Create structured output
                output = []
                
                # Add inventory overview section
                output.append("# Inventory Analysis")
                
                # Add summary metrics
                output.append("\n## Summary Metrics")
                critical_count = sum(1 for i in items if int(i.get('CurrentStock', '0').replace(',', '')) < 5000)
                low_count = sum(1 for i in items if 5000 <= int(i.get('CurrentStock', '0').replace(',', '')) < 20000)
                moderate_count = sum(1 for i in items if 20000 <= int(i.get('CurrentStock', '0').replace(',', '')) < 100000)
                healthy_count = sum(1 for i in items if int(i.get('CurrentStock', '0').replace(',', '')) >= 100000)
                
                output.append("| Metric | Count |")
                output.append("|--------|-------|")
                output.append(f"| Total Items | {len(items)} |")
                output.append(f"| Critical Stock (<5,000) | {critical_count} |")
                output.append(f"| Low Stock (5,000-19,999) | {low_count} |")
                output.append(f"| Moderate Stock (20,000-99,999) | {moderate_count} |")
                output.append(f"| Healthy Stock (≥100,000) | {healthy_count} |")
                
                # Add detailed inventory table
                output.append("\n## Detailed Inventory Status")
                
                # Create table headers
                headers = [
                    'Item Name', 'Current Stock', 'Status',
                    'Stock Level %', 'Lead Time', 'Action Required'
                ]
                
                # Create markdown table
                output.append("| " + " | ".join(headers) + " |")
                output.append("|" + "|".join(["---" for _ in headers]) + "|")
                
                # Sort items by stock level (ascending) to prioritize critical items
                items.sort(key=lambda x: int(x.get('CurrentStock', '0').replace(',', '')))
                
                # Add rows with enhanced formatting
                for item in items:
                    current_stock = int(item.get('CurrentStock', '0').replace(',', ''))
                    max_inventory = int(item.get('MaxInventory', '100000').replace(',', ''))
                    stock_level = (current_stock / max_inventory * 100) if max_inventory > 0 else 0
                    
                    # Determine status and action
                    if current_stock < 5000:
                        status = "🔴 Critical"
                        action = "Immediate Reorder"
                    elif current_stock < 20000:
                        status = "🟡 Low"
                        action = "Plan Reorder"
                    elif current_stock < 100000:
                        status = "🟦 Moderate"
                        action = "Monitor"
                    else:
                        status = "🟢 Healthy"
                        action = "None Required"
                    
                    row = [
                        item.get('GenericName', 'N/A'),
                        f"{current_stock:,}",
                        status,
                        f"{stock_level:.1f}%",
                        f"{item.get('LeadTime', 'N/A')} days",
                        action
                    ]
                    output.append("| " + " | ".join(str(cell) for cell in row) + " |")
                
                # Add critical items section if any exist
                if critical_count > 0:
                    output.append("\n## Critical Items Requiring Immediate Attention")
                    critical_items = [item for item in items if int(item.get('CurrentStock', '0').replace(',', '')) < 5000]
                    for item in critical_items:
                        current_stock = int(item.get('CurrentStock', '0').replace(',', ''))
                        output.append(f"• {item.get('GenericName', 'N/A')}: {current_stock:,} units remaining")
                
                return "\n".join(output)
            
            return text
            
        except Exception as e:
            logger.error(f"Error formatting tabular data: {str(e)}")
            return text
    
    def _generate_charts(self, text: str) -> list:
        """Generate visualizations for the analysis"""
        try:
            # Extract data for visualization
            items = []
            for line in text.split('\n'):
                if 'CurrentStock' in line:
                    name_match = re.search(r'([A-Za-z\s]+)(?=\s*[:|]\s*CurrentStock)', line)
                    stock_match = re.search(r'CurrentStock\s*[-:]\s*([\d,]+)', line)
                    if name_match and stock_match:
                        items.append({
                            'name': name_match.group(1).strip(),
                            'stock': int(stock_match.group(1).replace(',', ''))
                        })
            
            if not items:
                return []
            
            charts = []
            
            # Create stock level status chart
            plt.figure(figsize=(12, 6))
            
            # Sort items by stock level
            items.sort(key=lambda x: x['stock'])
            names = [item['name'] for item in items]
            stocks = [item['stock'] for item in items]
            
            # Create horizontal bar chart
            colors = ['red' if s < 5000 else 'yellow' if s < 20000 else 'blue' if s < 100000 else 'green' for s in stocks]
            plt.barh(names, stocks, color=colors)
            
            plt.title('Stock Level Status by Item')
            plt.xlabel('Current Stock')
            
            # Add value labels
            for i, v in enumerate(stocks):
                plt.text(v, i, f' {v:,}', va='center')
            
            # Save chart
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close()
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            charts.append({
                'title': 'Stock Level Status',
                'type': 'bar',
                'image': image_base64,
                'description': 'Horizontal bar chart showing current stock levels for all items'
            })
            
            # Create stock level distribution pie chart
            plt.figure(figsize=(8, 8))
            
            # Calculate distribution
            critical = sum(1 for s in stocks if s < 5000)
            low = sum(1 for s in stocks if 5000 <= s < 20000)
            moderate = sum(1 for s in stocks if 20000 <= s < 100000)
            healthy = sum(1 for s in stocks if s >= 100000)
            
            sizes = [critical, low, moderate, healthy]
            labels = ['Critical', 'Low', 'Moderate', 'Healthy']
            colors = ['red', 'yellow', 'blue', 'green']
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            plt.title('Stock Level Distribution')
            
            # Save chart
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close()
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            charts.append({
                'title': 'Stock Level Distribution',
                'type': 'pie',
                'image': image_base64,
                'description': 'Pie chart showing distribution of items across stock level categories'
            })
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            return [] 