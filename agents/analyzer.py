import re
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import AppConfig
import logging
import os

logger = logging.getLogger(__name__)

class TransportAnalyzer:
    """Analyzer specifically for transport history data"""
    def __init__(self):
        self.config = AppConfig()
        self.llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=self.config.OPENAI_MODEL,
            temperature=0.7
        )
        self.transport_prompt = """
        As a logistics and transportation analyst in the pharmaceutical supply chain, analyze the following transport history data.
        Provide insights in a clear, professional, and actionable manner.

        Analysis Requirements:

        1. Delivery Performance:
           - On-time delivery rates
           - Average transit times
           - Delivery success rates
           - Delay patterns and causes

        2. Route Analysis:
           - Most frequent routes
           - Route efficiency metrics
           - Geographic distribution
           - Optimization opportunities

        3. Carrier Performance:
           - Carrier reliability ratings
           - Cost-efficiency analysis
           - Service quality metrics
           - Carrier comparison

        4. Temperature Control:
           - Temperature compliance rates
           - Temperature excursion incidents
           - Cold chain integrity
           - Risk areas identified

        5. Cost Analysis:
           - Transport cost per route
           - Cost trends and patterns
           - Cost optimization opportunities
           - Budget implications

        6. Risk Assessment:
           - High-risk routes identified
           - Common issues and challenges
           - Mitigation strategies
           - Safety and compliance status

        7. Recommendations:
           - Route optimization suggestions
           - Carrier selection guidance
           - Cost reduction opportunities
           - Quality improvement measures

        Data to Analyze:
        {data}

        Provide a professional analysis with clear sections and actionable insights.
        Use natural language and include specific metrics and percentages where relevant.
        Focus on patterns, trends, and opportunities for improvement.
        """
        self.prompt = ChatPromptTemplate.from_template(self.transport_prompt)

    def analyze_transport(self, data: str) -> dict:
        """Analyze transport history data"""
        try:
            # Format data in tabular form
            formatted_data = self._format_transport_data(data)
            
            # Get analysis from LLM
            chain = self.prompt | self.llm
            analysis = chain.invoke({"data": formatted_data})
            
            # Generate visualizations
            charts = self._generate_transport_charts(data)
            
            return {
                'insights': analysis.content,
                'formatted_response': formatted_data,
                'charts': charts,
                'has_numerical_data': True
            }
        except Exception as e:
            logger.error(f"Error in transport analysis: {str(e)}")
            return {
                'insights': "Error analyzing transport data. Raw data:\n\n" + data,
                'formatted_response': data,
                'charts': [],
                'has_numerical_data': False
            }

    def _format_transport_data(self, text: str) -> str:
        """Format transport data in a tabular structure"""
        try:
            # Extract transport records using regex
            records = []
            
            # Enhanced patterns for transport data
            patterns = {
                'ShipmentID': r'SHIP-[A-Z0-9]+',
                'Origin': r'Origin:\s*([^,\n]+)',
                'Destination': r'Destination:\s*([^,\n]+)',
                'CarrierName': r'Carrier:\s*([^,\n]+)',
                'TransitTime': r'TransitTime:\s*(\d+)',
                'Temperature': r'Temperature:\s*([-\d.]+)',
                'Status': r'Status:\s*([^,\n]+)',
                'Cost': r'Cost:\s*\$?([\d,.]+)',
                'DeliveryDate': r'DeliveryDate:\s*([^,\n]+)'
            }
            
            # Parse records
            record_blocks = text.split('\n\n')
            for block in record_blocks:
                record = {}
                for key, pattern in patterns.items():
                    match = re.search(pattern, block)
                    if match:
                        record[key] = match.group(1).strip()
                if record:
                    records.append(record)
            
            if records:
                # Create table headers
                headers = [
                    'Shipment ID', 'Origin', 'Destination', 'Transit Time',
                    'Temperature', 'Status', 'Cost', 'Carrier'
                ]
                
                # Create markdown table
                table = "| " + " | ".join(headers) + " |\n"
                table += "|" + "|".join(["---" for _ in headers]) + "|\n"
                
                # Add rows
                for record in records:
                    row = [
                        record.get('ShipmentID', 'N/A'),
                        record.get('Origin', 'N/A'),
                        record.get('Destination', 'N/A'),
                        f"{record.get('TransitTime', 'N/A')} days",
                        f"{record.get('Temperature', 'N/A')}°C",
                        self._get_status_emoji(record.get('Status', 'Unknown')),
                        f"${record.get('Cost', 'N/A')}",
                        record.get('CarrierName', 'N/A')
                    ]
                    table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
                
                # Add summary
                summary = "\n### Transport Summary\n"
                summary += f"- Total Shipments: {len(records)}\n"
                summary += f"- On-Time Deliveries: {sum(1 for r in records if 'On Time' in r.get('Status', ''))}\n"
                summary += f"- Average Transit Time: {sum(float(r.get('TransitTime', 0)) for r in records) / len(records):.1f} days\n"
                summary += f"- Total Transport Cost: ${sum(float(r.get('Cost', 0)) for r in records):,.2f}\n"
                
                return f"{table}\n{summary}\n\n### Detailed Analysis\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error formatting transport data: {str(e)}")
            return text

    def _get_status_emoji(self, status: str) -> str:
        """Get appropriate emoji for transport status"""
        status = status.lower()
        if 'on time' in status:
            return "✅ On Time"
        elif 'delay' in status:
            return "⚠️ Delayed"
        elif 'cancel' in status:
            return "❌ Cancelled"
        elif 'transit' in status:
            return "🚚 In Transit"
        return status

    def _generate_transport_charts(self, data: str) -> list:
        """Generate charts for transport analysis"""
        try:
            # Extract data for visualization
            records = self._extract_transport_data(data)
            if not records:
                return []
            
            charts = []
            
            # 1. Transit Time Distribution
            plt.figure(figsize=(12, 6), facecolor='#f0f0f0')
            transit_times = [float(r.get('TransitTime', 0)) for r in records]
            plt.hist(transit_times, bins=20, color='#3498db', alpha=0.7)
            plt.xlabel('Transit Time (days)')
            plt.ylabel('Number of Shipments')
            plt.title('Distribution of Transit Times')
            plt.grid(True, alpha=0.3)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                      facecolor='#f0f0f0', edgecolor='none')
            buffer.seek(0)
            charts.append({
                'type': 'histogram',
                'data': base64.b64encode(buffer.getvalue()).decode(),
                'title': 'Transit Time Distribution',
                'description': 'Shows the distribution of shipment transit times'
            })
            plt.close()
            
            # 2. Status Distribution Pie Chart
            plt.figure(figsize=(10, 10), facecolor='#f0f0f0')
            status_counts = {}
            for record in records:
                status = record.get('Status', 'Unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            plt.pie(
                status_counts.values(),
                labels=status_counts.keys(),
                autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c', '#f1c40f', '#3498db'],
                startangle=90
            )
            plt.title('Distribution of Shipment Status')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                      facecolor='#f0f0f0', edgecolor='none')
            buffer.seek(0)
            charts.append({
                'type': 'pie',
                'data': base64.b64encode(buffer.getvalue()).decode(),
                'title': 'Shipment Status Distribution',
                'description': 'Shows the distribution of shipment statuses'
            })
            plt.close()
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating transport charts: {str(e)}")
            return []

    def _extract_transport_data(self, text: str) -> list:
        """Extract structured data from transport text"""
        try:
            records = []
            patterns = {
                'ShipmentID': r'SHIP-[A-Z0-9]+',
                'TransitTime': r'TransitTime:\s*(\d+)',
                'Temperature': r'Temperature:\s*([-\d.]+)',
                'Status': r'Status:\s*([^,\n]+)',
                'Cost': r'Cost:\s*\$?([\d,.]+)'
            }
            
            record_blocks = text.split('\n\n')
            for block in record_blocks:
                record = {}
                for key, pattern in patterns.items():
                    match = re.search(pattern, block)
                    if match:
                        record[key] = match.group(1).strip()
                if record:
                    records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Error extracting transport data: {str(e)}")
            return []

class AnalyticalAgent:
    def __init__(self):
        self.config = AppConfig()
        self.llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=self.config.OPENAI_MODEL,
            temperature=0.7
        )
        # Enhanced analysis prompt for better insights
        self.analysis_prompt = """
        As an expert pharmaceutical inventory analyst, provide a comprehensive analysis of the following data.
        Focus on delivering actionable insights in a professional yet engaging manner.

        Required Analysis Components:

        1. Executive Summary:
           - Brief overview of inventory status
           - Key highlights and critical findings
           - Overall supply chain health assessment

        2. Critical Stock Analysis:
           - Items with critically low stock (<5,000 units)
           - Immediate action requirements
           - Impact assessment on patient care
           - Risk mitigation strategies

        3. Inventory Health Categories:
           - Critical: <5,000 units
           - Low: 5,000-20,000 units
           - Moderate: 20,000-100,000 units
           - Healthy: >100,000 units
           - Percentage distribution across categories
           - Trend analysis where possible

        4. Supply Chain Risk Assessment:
           - High-risk items requiring immediate attention
           - Medium-risk items needing monitoring
           - Low-risk items with stable supply
           - Potential disruption impacts
           - Mitigation strategies

        5. Cost and Efficiency Analysis:
           - Estimated reordering costs
           - Storage optimization opportunities
           - Budget implications
           - Efficiency improvement recommendations

        6. Action Plan:
           - Immediate actions (24-48 hours)
           - Short-term actions (1 week)
           - Medium-term planning (1 month)
           - Long-term strategic recommendations

        7. Compliance and Quality:
           - Regulatory compliance status
           - Quality control measures
           - Storage condition monitoring
           - Expiration date management

        Data to Analyze:
        {response}

        Format the response professionally with clear sections, bullet points where appropriate, and specific recommendations.
        Use natural language and maintain a friendly yet authoritative tone.
        Include specific numbers and percentages to support your analysis.
        """
        self.prompt = ChatPromptTemplate.from_template(self.analysis_prompt)
        self.transport_analyzer = TransportAnalyzer()
        
    def analyze(self, response: str) -> dict:
        """
        Analyze the response and generate insights with visualizations.
        Handles both inventory and transport history data.
        """
        try:
            # Determine if this is transport history data
            if 'SHIP-' in response or 'Transit' in response:
                return self.transport_analyzer.analyze_transport(response)
            
            # Otherwise, proceed with inventory analysis
            formatted_response = self._format_tabular_data(response)
            chain = self.prompt | self.llm
            analysis = chain.invoke({"response": formatted_response})
            
            data = self._extract_numerical_data(response)
            charts = []
            
            if all(len(d) > 0 for d in data):
                bar_chart = self._generate_chart(data, None, 'bar')
                if bar_chart:
                    charts.append({
                        'type': 'bar',
                        'data': bar_chart,
                        'title': 'Inventory Levels Comparison'
                    })
                
                pie_chart = self._generate_chart(data, None, 'pie')
                if pie_chart:
                    charts.append({
                        'type': 'pie',
                        'data': pie_chart,
                        'title': 'Stock Level Distribution'
                    })
            
            return {
                'insights': analysis.content,
                'formatted_response': formatted_response,
                'charts': charts,
                'has_numerical_data': bool(data[0])
            }
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {
                'insights': "Error analyzing the response. The raw response is:\n\n" + response,
                'formatted_response': response,
                'charts': [],
                'has_numerical_data': False
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

    def _extract_numerical_data(self, text):
        """Extract numerical data from text for visualization"""
        try:
            # Find all numbers and their labels in the text
            pattern = r'(\w+(?:\s+\w+)*)\s*:\s*(\d+(?:,\d+)*(?:\.\d+)?)'
            matches = re.findall(pattern, text)
            
            # Process matches into separate lists
            data_dict = {}
            
            for label, number in matches:
                try:
                    # Clean up the number (remove commas)
                    clean_number = float(number.replace(',', ''))
                    
                    # Only add if it's a valid number and not an ID
                    label = label.strip()
                    if not (label.lower().startswith('itemid') or 
                           label.lower().startswith('batch') or
                           label.lower().startswith('unit') or
                           label.lower().startswith('lead')):
                        if label not in data_dict:
                            data_dict[label] = clean_number
                except ValueError as e:
                    logger.warning(f"Could not convert number '{number}' to float: {str(e)}")
                    continue
            
            # Extract product names and their corresponding values
            products = []
            max_inventory = []
            current_stock = []
            
            # Group data by product
            current_product = None
            for label, value in data_dict.items():
                if label == 'MaxInventory':
                    if current_product:
                        products.append(current_product)
                    current_product = None
                elif label == 'GenericName':
                    current_product = value
                elif label == 'CurrentStock' and current_product:
                    current_stock.append(value)
                    max_inventory.append(data_dict.get('MaxInventory', 0))
                    
            return max_inventory, current_stock, products
            
        except Exception as e:
            logger.error(f"Error extracting numerical data: {str(e)}")
            return [], [], []
        
    def _generate_chart(self, data, labels, chart_type='bar'):
        """Generate a chart using matplotlib"""
        try:
            max_inventory, current_stock, products = data
            
            if not max_inventory or not current_stock or not products:
                logger.warning("Insufficient data for chart generation")
                plt.close()
                return None
            
            if len(max_inventory) != len(current_stock) or len(current_stock) != len(products):
                logger.error(f"Data length mismatch: max_inventory({len(max_inventory)}), current_stock({len(current_stock)}), products({len(products)})")
                plt.close()
                return None
            
            # Create multiple charts for different aspects of the inventory
            charts = []
            
            # 1. Stock Level Status Chart
            plt.figure(figsize=(15, 8), facecolor='#f0f0f0')
            
            # Calculate stock levels and categorize
            stock_levels = []
            categories = []
            colors = []
            
            for curr in current_stock:
                if curr < 5000:
                    categories.append('Critical')
                    colors.append('#e74c3c')  # Red
                elif curr < 20000:
                    categories.append('Low')
                    colors.append('#f1c40f')  # Yellow
                elif curr < 100000:
                    categories.append('Moderate')
                    colors.append('#3498db')  # Blue
                else:
                    categories.append('Healthy')
                    colors.append('#2ecc71')  # Green
                stock_levels.append(curr)
            
            # Create horizontal bar chart
            y_pos = range(len(products))
            plt.barh(y_pos, stock_levels, color=colors, alpha=0.7)
            plt.yticks(y_pos, products)
            plt.xlabel('Current Stock')
            plt.title('Inventory Status by Product', pad=20)
            
            # Add value labels
            for i, v in enumerate(stock_levels):
                plt.text(v, i, f' {v:,}', va='center')
            
            # Add color-coded legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#e74c3c', label='Critical (<5,000)', alpha=0.7),
                Patch(facecolor='#f1c40f', label='Low (<20,000)', alpha=0.7),
                Patch(facecolor='#3498db', label='Moderate (<100,000)', alpha=0.7),
                Patch(facecolor='#2ecc71', label='Healthy (≥100,000)', alpha=0.7)
            ]
            plt.legend(handles=legend_elements, loc='lower right')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save first chart
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                      facecolor='#f0f0f0', edgecolor='none')
            buffer.seek(0)
            charts.append({
                'type': 'bar',
                'data': base64.b64encode(buffer.getvalue()).decode(),
                'title': 'Inventory Status by Product'
            })
            plt.close()
            
            # 2. Stock Level Distribution Pie Chart
            plt.figure(figsize=(10, 10), facecolor='#f0f0f0')
            
            # Count items in each category
            category_counts = {
                'Critical': categories.count('Critical'),
                'Low': categories.count('Low'),
                'Moderate': categories.count('Moderate'),
                'Healthy': categories.count('Healthy')
            }
            
            # Create pie chart
            colors = ['#e74c3c', '#f1c40f', '#3498db', '#2ecc71']
            plt.pie(
                category_counts.values(),
                labels=category_counts.keys(),
                colors=colors,
                autopct='%1.1f%%',
                startangle=90
            )
            plt.title('Distribution of Stock Levels', pad=20)
            
            # Save second chart
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                      facecolor='#f0f0f0', edgecolor='none')
            buffer.seek(0)
            charts.append({
                'type': 'pie',
                'data': base64.b64encode(buffer.getvalue()).decode(),
                'title': 'Distribution of Stock Levels'
            })
            plt.close()
            
            return charts
            
        except Exception as e:
            logger.error(f"Error in chart generation: {str(e)}")
            if plt.get_fignums():
                plt.close()
            return None 