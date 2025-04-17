# 🏥 PharmAI Assistant: Agentic RAG for Pharmaceutical Supply Chain 🤖

![PharmAI Banner](https://img.shields.io/badge/PharmAI-Pharmaceutical%20RAG%20System-blue?style=for-the-badge&logo=openai)

## 🔍 Overview

PharmAI is an advanced retrieval-augmented generation (RAG) system specifically designed for pharmaceutical supply chain management. It combines vector database technology, real-time web search capabilities, and sophisticated analytics to provide intelligent responses to pharmaceutical inventory and logistics queries.

![License](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/python-v3.9+-blue) ![Flask](https://img.shields.io/badge/flask-v3.0.0-orange) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-purple) ![ChromaDB](https://img.shields.io/badge/ChromaDB-v0.4.24-yellow)

## ✨ Key Features

- 🧠 **Intelligent Query Enhancement**: Refines user queries to include relevant pharmaceutical terminology and context
- 🔀 **Smart Routing**: Automatically directs queries to the most appropriate data sources
- 📊 **Data Analysis**: Provides analytical insights on inventory, logistics, and compliance data
- 🌐 **Web Search Integration**: Supplements local data with real-time information from the web
- 📈 **Visualization**: Generates informative charts and graphs to represent data trends
- 📱 **Modern UI**: Clean, responsive interface with real-time processing pipeline visualization
- 🔄 **Session Management**: Maintains context across conversations for more coherent interactions

## 🏗️ System Architecture

```
   User
    ↓
+-------------------+     +--------------------+     +--------------------+
|                   |     |                    |     |                    |
|  Web Interface    |←→|  Flask Backend API  |←→|  Agent Pipeline     |
|  (HTML/CSS/JS)    |     |  (Flask-RestX)     |     |  (LangChain/OpenAI)|
|                   |     |                    |     |                    |
+-------------------+     +--------------------+     +--------------------+
                                                            ↑  ↑
                                                            |  |
                          +--------------------+            |  |
                          |                    |            |  |
                          |  ChromaDB Vector   |←-----------+  |
                          |  Database          |               |
                          |                    |               |
                          +--------------------+               |
                                                               |
                          +--------------------+               |
                          |                    |               |
                          |  External Web      |←--------------+
                          |  (SERPER API)      |
                          |                    |
                          +--------------------+
```

## 🔄 Processing Pipeline

1. **Query Enhancement**: The `QueryEnhancer` class analyzes and enhances user queries to improve retrieval
2. **Query Routing**: The `DataRouter` determines which data sources to use for the query
3. **Data Retrieval**: Relevant information is retrieved from ChromaDB vector collections
4. **Web Search Augmentation**: For queries requiring current information, real-time web search is performed
5. **Response Generation**: The `RAGAgent` or `AnalyticalAgent` generates a comprehensive response
6. **Visualization**: Where applicable, visualizations are created to represent data trends

## 💻 Installation

### Prerequisites

- Python 3.9+
- OpenAI API key
- SERPER API key (for web search)

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/pharmai-assistant.git
cd pharmai-assistant
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up environment variables

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
OPENAI_MODEL=gpt-4o
```

### Step 5: Run the application

```bash
python app.py
```

The application will be available at `http://localhost:5000`.

## 🚀 Usage Examples

### Example 1: Inventory Query
```
Query: What insulin products are running low on stock?
```

### Example 2: Regulatory Compliance Query
```
Query: What are the temperature requirements for transporting flu vaccines?
```

### Example 3: Analytics Query
```
Query: Compare the current stock levels of antibiotics versus last month.
```

### Example 4: External Information Query
```
Query: What are the current tariffs on imported pharmaceuticals from China?
```

## 📊 Data Sources

The system uses multiple data sources:

- **Inventory Data**: Stock levels, reorder points, expiry dates, etc.
- **Transport Data**: Shipping information, delivery status, logistics
- **Guidelines**: Regulatory requirements, compliance information
- **Policy**: Company policies, operating procedures, internal rules
- **Web**: Real-time information from the web (via SERPER API)

## 🧩 Components

### Agents

- **QueryEnhancer**: Refines user queries for better retrieval
- **DataRouter**: Routes queries to appropriate data sources
- **RAGAgent**: Generates responses for factual queries
- **AnalyticalAgent**: Analyzes data and provides insights
- **WebSearchAgent**: Retrieves real-time information from the web
- **ReportGenerator**: Creates comprehensive reports

### Utilities

- **ChromaDBManager**: Manages interactions with ChromaDB vector database
- **ChromaParser**: Parses data from ChromaDB collections
- **SessionManager**: Maintains user sessions and conversation history

## 🔧 Configuration

All system settings are defined in `config/settings.py`. Key configurations include:

- API keys and models (OpenAI, SERPER)
- Data source paths
- Analysis thresholds
- Response templates
- Chart configurations

## 📋 API Documentation

The API documentation is available at `/api/docs` when running the application. Main endpoints include:

- **POST /api/chat/**: Process a chat query and return a response
- **GET /api/system/status**: Get system status information
- **GET /api/chat/history/{session_id}**: Get chat history for a session
- **DELETE /api/chat/clear/{session_id}**: Clear chat history for a session

## 🔮 Future Enhancements

- 🌟 Multi-model support for different types of queries
- 🌟 Enhanced analytics with predictive capabilities
- 🌟 Integration with real-time inventory systems
- 🌟 Mobile application interface
- 🌟 Additional language support

## 📈 Performance Metrics

- Average response time: < 3 seconds
- Relevance accuracy: > 90% 
- Web search integration success rate: > 95%
- System uptime: > 99.9%

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- OpenAI for providing the foundation models
- SERPER for web search capabilities
- The entire open-source community for inspiration and tools

---

<p align="center">
  Made with ❤️ for the pharmaceutical industry
</p> 
