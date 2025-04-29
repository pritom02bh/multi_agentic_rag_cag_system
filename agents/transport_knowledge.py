import os
import json
import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class TransportKnowledgeBase:
    """Knowledge base for transport-related information."""
    
    def __init__(self, knowledge_file: str = 'embedded_data/transport_knowledge_base.json'):
        """Initialize the knowledge base from a JSON file."""
        self.knowledge_file = knowledge_file
        self.rules = [
            {
                "id": "TEMP_RULE_1",
                "description": "Temperature-sensitive medications must be transported in validated cold chain containers.",
                "applies_to": ["insulin", "vaccines", "biologics", "flu vaccine", "covid-19 vaccine"]
            },
            {
                "id": "DOC_RULE_1",
                "description": "All shipments must have complete documentation including temperature logs.",
                "applies_to": ["all"]
            },
            {
                "id": "REG_RULE_1",
                "description": "Controlled substances require DEA Form 222 for transfers.",
                "applies_to": ["controlled substances"]
            },
            {
                "id": "SAFETY_RULE_1",
                "description": "Hazardous materials must be transported according to PHMSA regulations.",
                "applies_to": ["hazardous", "flammable", "corrosive"]
            },
            {
                "id": "GEN_RULE_1",
                "description": "All pharmaceutical shipments must maintain product integrity throughout transit.",
                "applies_to": ["all"]
            }
        ]
        
        self.medicine_data = {
            "insulin": {
                "temperature_range": "2-8°C",
                "storage_requirements": "Refrigerated",
                "shipment_performance": {
                    "avg_arrival_delay": 0.5,
                    "temperature_excursions": "Low"
                },
                "routes": {
                    "carriers": ["MedExpress", "ColdChain Logistics", "MedAir"]
                },
                "risk_assessment": {
                    "category": "High Risk",
                    "factors": ["Temperature sensitive", "High value"]
                },
                "environmental_impact": {
                    "carbon_footprint": "Medium",
                    "packaging_waste": "Low (Reusable containers)"
                }
            },
            "flu vaccine": {
                "temperature_range": "2-8°C",
                "storage_requirements": "Refrigerated",
                "shipment_performance": {
                    "avg_arrival_delay": 0.3,
                    "temperature_excursions": "Very Low"
                },
                "routes": {
                    "carriers": ["VaxFreight", "ColdChain Logistics", "MedExpress"]
                },
                "risk_assessment": {
                    "category": "High Risk",
                    "factors": ["Temperature sensitive", "Seasonal demand"]
                },
                "environmental_impact": {
                    "carbon_footprint": "Medium-High",
                    "packaging_waste": "Medium (Specialized containers)"
                }
            },
            "paracetamol": {
                "temperature_range": "15-25°C",
                "storage_requirements": "Room temperature, dry place",
                "shipment_performance": {
                    "avg_arrival_delay": 0.2,
                    "temperature_excursions": "Very Low"
                },
                "routes": {
                    "carriers": ["MedFreight", "GlobalRx", "PharmaShip"]
                },
                "risk_assessment": {
                    "category": "Low Risk",
                    "factors": ["Temperature stable", "Common medication"]
                },
                "environmental_impact": {
                    "carbon_footprint": "Low",
                    "packaging_waste": "Medium (Standard packaging)"
                }
            }
        }
        
        # Try to load knowledge base from file if it exists
        try:
            if os.path.exists(knowledge_file):
                with open(knowledge_file, 'r') as f:
                    data = json.load(f)
                    if 'rules' in data:
                        self.rules = data['rules']
                    if 'medicines' in data:
                        self.medicine_data.update(data['medicines'])
                logger.info(f"Loaded transport knowledge base from {knowledge_file}")
            else:
                logger.warning(f"Transport knowledge file not found: {knowledge_file}")
        except Exception as e:
            logger.error(f"Error loading transport knowledge base: {str(e)}")
        
        logger.info(f"TransportKnowledgeBase initialized, loaded {len(self.rules)} rules")
    
    def get_medicine_info(self, medicine_name: str) -> Optional[Dict]:
        """
        Get information about a specific medicine.
        
        Args:
            medicine_name: Name of the medicine
            
        Returns:
            Dictionary of medicine information or None
        """
        # Convert to lowercase for case-insensitive matching
        medicine_name_lower = medicine_name.lower()
        
        # Check for exact match
        if medicine_name_lower in self.medicine_data:
            return self.medicine_data[medicine_name_lower]
        
        # Check for partial match
        for name, info in self.medicine_data.items():
            if medicine_name_lower in name.lower() or name.lower() in medicine_name_lower:
                return info
        
        return None
    
    def get_rules_for_medicine(self, medicine_name: str) -> List[Dict]:
        """
        Get applicable transport rules for a specific medicine.
        
        Args:
            medicine_name: Name of the medicine
            
        Returns:
            List of applicable rule dictionaries
        """
        applicable_rules = []
        medicine_name_lower = medicine_name.lower()
        
        for rule in self.rules:
            applies_to = rule.get('applies_to', [])
            
            # Rule applies to all medicines
            if 'all' in applies_to:
                applicable_rules.append(rule)
                continue
                
            # Check if rule applies to this specific medicine
            if any(medicine_term.lower() in medicine_name_lower or medicine_name_lower in medicine_term.lower() for medicine_term in applies_to):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def get_relevant_knowledge(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge relevant to a transport query.
        
        Args:
            query: The query text
            
        Returns:
            Dictionary with relevant knowledge or None
        """
        query_lower = query.lower()
        
        # Find medicines mentioned in the query
        medicines = []
        for medicine_name in self.medicine_data.keys():
            if medicine_name.lower() in query_lower:
                medicines.append(medicine_name)
        
        # Identify keywords in the query
        keywords = {
            "temperature": ["temperature", "cold chain", "cooling", "refrigerat", "frozen"],
            "documentation": ["document", "paperwork", "form", "certificate", "log"],
            "regulation": ["regulat", "compliance", "fda", "phmsa", "rule", "guideline"],
            "carrier": ["carrier", "transport", "shipping", "delivery", "logistics"],
            "safety": ["safety", "secure", "protection", "integrity", "damage"],
            "sustainability": ["environment", "carbon", "footprint", "sustainable", "green"]
        }
        
        # Match keywords
        matched_categories = {}
        for category, terms in keywords.items():
            for term in terms:
                if term in query_lower:
                    matched_categories[category] = matched_categories.get(category, 0) + 1
        
        # No matches found
        if not medicines and not matched_categories:
            return None
        
        # Compile relevant knowledge
        knowledge = {
            "medicines": {},
            "general_rules": [],
            "query_focus": list(matched_categories.keys())
        }
        
        # Add medicine-specific information
        for medicine in medicines:
            medicine_info = self.get_medicine_info(medicine)
            if medicine_info:
                knowledge["medicines"][medicine] = medicine_info
            
            # Add rules for this medicine
            medicine_rules = self.get_rules_for_medicine(medicine)
            for rule in medicine_rules:
                if rule not in knowledge["general_rules"]:
                    knowledge["general_rules"].append(rule)
        
        # If no specific medicines found, add general rules
        if not medicines:
            for rule in self.rules:
                if 'all' in rule.get('applies_to', []):
                    knowledge["general_rules"].append(rule)
        
        # Add transport guidelines based on matched categories
        if matched_categories:
            knowledge["guidelines"] = {}
            
            if "temperature" in matched_categories:
                knowledge["guidelines"]["temperature"] = [
                    "Temperature-sensitive products must be shipped in qualified containers",
                    "Temperature monitoring devices must be included in shipments",
                    "Temperature excursions must be documented and investigated"
                ]
            
            if "documentation" in matched_categories:
                knowledge["guidelines"]["documentation"] = [
                    "All shipments must include complete documentation",
                    "Temperature logs must be maintained for cold chain products",
                    "Chain of custody must be documented for controlled substances"
                ]
            
            if "regulation" in matched_categories:
                knowledge["guidelines"]["regulation"] = [
                    "FDA requires proper storage and handling of pharmaceuticals during transport",
                    "PHMSA regulates transport of hazardous materials including certain pharmaceuticals",
                    "International Air Transport Association (IATA) has specific requirements for air transport"
                ]
            
            if "carrier" in matched_categories:
                knowledge["guidelines"]["carrier"] = [
                    "Carriers must be qualified and approved for pharmaceutical transport",
                    "Specialized carriers should be used for temperature-sensitive products",
                    "Carrier performance should be regularly monitored and evaluated"
                ]
            
            if "safety" in matched_categories:
                knowledge["guidelines"]["safety"] = [
                    "Products must be secured to prevent damage during transit",
                    "Safety data sheets must accompany hazardous materials",
                    "Appropriate packaging must be used to maintain product integrity"
                ]
            
            if "sustainability" in matched_categories:
                knowledge["guidelines"]["sustainability"] = [
                    "Reusable shipping containers should be used when possible",
                    "Route optimization reduces carbon footprint",
                    "Carriers with sustainability initiatives should be preferred"
                ]
        
        return knowledge
    
    def get_knowledge(self) -> str:
        """Get knowledge as formatted string for context."""
        try:
            if not self.rules:
                return "No transport knowledge available."
            
            knowledge_str = "PHARMACEUTICAL TRANSPORT KNOWLEDGE:\n\n"
            
            # Rules
            knowledge_str += "RULES:\n"
            for rule in self.rules:
                knowledge_str += f"- {rule['id']}: {rule['description']}\n"
            
            # Medicine data
            knowledge_str += "\nMEDICINE DATA:\n"
            for medicine, info in self.medicine_data.items():
                knowledge_str += f"- {medicine}:\n"
                for key, value in info.items():
                    if isinstance(value, dict):
                        knowledge_str += f"  - {key}:\n"
                        for subkey, subvalue in value.items():
                            knowledge_str += f"    - {subkey}: {subvalue}\n"
                    else:
                        knowledge_str += f"  - {key}: {value}\n"
            
            return knowledge_str
            
        except Exception as e:
            logger.error(f"Error formatting transport knowledge: {str(e)}")
            return "Error retrieving transport knowledge."
            
    def generate_transport_insights(self, query: str) -> str:
        """Generate transport insights based on a query.
        
        This is an alias for get_knowledge() for compatibility with newer code.
        
        Args:
            query: The user's query (not used in this implementation)
            
        Returns:
            str: Transport knowledge as formatted string
        """
        return self.get_knowledge()
    
    def get_temperature_requirement(self, product_type: str) -> Optional[str]:
        """Get temperature requirement for a product type."""
        try:
            product_type = product_type.lower()
            
            # Direct lookup in temperature requirements
            temp_req = self.medicine_data.get(product_type, {}).get("temperature_range")
            if temp_req:
                return temp_req
            
            # Check product classifications
            classification = self.rules[0].get("applies_to", [])[0] if product_type in self.rules[0].get("applies_to", []) else None
            if classification:
                return self.medicine_data.get(classification, {}).get("temperature_range")
            
            return None
        except Exception as e:
            logger.error(f"Error getting temperature requirement: {str(e)}")
            return None
    
    def get_carrier_rating(self, carrier_name: str) -> Optional[float]:
        """Get quality rating for a carrier."""
        try:
            carrier_ratings = self.medicine_data.get(carrier_name, {})
            return carrier_ratings.get("risk_assessment", {}).get("category")
        except Exception as e:
            logger.error(f"Error getting carrier rating: {str(e)}")
            return None
    
    def get_excursion_limit(self, product_type: str) -> Optional[str]:
        """Get excursion limit for a product type."""
        try:
            product_type = product_type.lower()
            
            # Direct lookup in excursion limits
            excursion_limit = self.medicine_data.get(product_type, {}).get("risk_assessment", {}).get("factors", [])[0] if "risk_assessment" in self.medicine_data.get(product_type, {}) else None
            if excursion_limit:
                return excursion_limit
            
            # Check product classifications
            classification = self.rules[0].get("applies_to", [])[0] if product_type in self.rules[0].get("applies_to", []) else None
            if classification:
                return self.medicine_data.get(classification, {}).get("risk_assessment", {}).get("factors", [])[0] if "risk_assessment" in self.medicine_data.get(classification, {}) else None
            
            return None
        except Exception as e:
            logger.error(f"Error getting excursion limit: {str(e)}")
            return None 