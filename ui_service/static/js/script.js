document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const welcomeScreen = document.getElementById('welcome-screen');
    const chatMessages = document.getElementById('chat-messages');
    const messagesContainer = document.getElementById('messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const newChatBtn = document.getElementById('new-chat-btn');
    const recentChats = document.getElementById('recent-chats');
    const examplePrompts = document.querySelectorAll('.example-prompt');

    // Pipeline stages
    const pipelineStages = {
        query: document.querySelector('[data-stage="query"]'),
        router: document.querySelector('[data-stage="router"]'),
        knowledge: document.querySelector('[data-stage="knowledge"]'),
        web: document.querySelector('[data-stage="web"]'),
        analytics: document.querySelector('[data-stage="analytics"]'),
        generator: document.querySelector('[data-stage="generator"]'),
        ui: document.querySelector('[data-stage="ui"]')
    };

    // Chat history management
    class ChatHistoryManager {
        constructor() {
            this.storageKey = 'pharmAI_chat_history';
            this.maxHistoryItems = 20;
            this.currentSessionId = this.generateSessionId();
            this.history = this.loadHistory();
        }

        generateSessionId() {
            return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        }

        loadHistory() {
            const stored = localStorage.getItem(this.storageKey);
            return stored ? JSON.parse(stored) : {};
        }

        saveHistory() {
            localStorage.setItem(this.storageKey, JSON.stringify(this.history));
        }

        addChat(message) {
            if (!this.history[this.currentSessionId]) {
                this.history[this.currentSessionId] = {
                    id: this.currentSessionId,
                    timestamp: new Date().toISOString(),
                    messages: []
                };
            }

            const chatItem = {
                id: Date.now(),
                content: message,
                timestamp: new Date().toISOString()
            };

            this.history[this.currentSessionId].messages.unshift(chatItem);
            if (this.history[this.currentSessionId].messages.length > this.maxHistoryItems) {
                this.history[this.currentSessionId].messages.pop();
            }
            this.saveHistory();
            return chatItem;
        }

        deleteChat(sessionId, messageId) {
            if (this.history[sessionId]) {
                this.history[sessionId].messages = this.history[sessionId].messages.filter(
                    chat => chat.id !== messageId
                );
                if (this.history[sessionId].messages.length === 0) {
                    delete this.history[sessionId];
                }
                this.saveHistory();
            }
        }

        deleteSession(sessionId) {
            if (this.history[sessionId]) {
                delete this.history[sessionId];
                this.saveHistory();
            }
        }

        clearHistory() {
            this.history = {};
            this.saveHistory();
        }

        getHistory() {
            return this.history;
        }

        startNewSession() {
            this.currentSessionId = this.generateSessionId();
            return this.currentSessionId;
        }
    }

    // Initialize chat history manager
    const chatHistoryManager = new ChatHistoryManager();

    // Add conversation context management
    class ConversationContext {
        constructor() {
            this.messages = [];
            this.lastQueryType = null;
            this.lastVisualization = null;
            this.inventoryContext = null;
        }

        addMessage(role, content, metadata = {}) {
            this.messages.push({
                role,
                content,
                timestamp: new Date(),
                metadata
            });
        }

        getLastUserMessage() {
            for (let i = this.messages.length - 1; i >= 0; i--) {
                if (this.messages[i].role === 'user') {
                    return this.messages[i];
                }
            }
            return null;
        }

        getLastAssistantMessage() {
            for (let i = this.messages.length - 1; i >= 0; i--) {
                if (this.messages[i].role === 'assistant') {
                    return this.messages[i];
                }
            }
            return null;
        }

        updateInventoryContext(data) {
            this.inventoryContext = {
                ...data,
                timestamp: new Date()
            };
        }

        updateVisualizationContext(data) {
            this.lastVisualization = {
                ...data,
                timestamp: new Date()
            };
        }

        isContextStale() {
            if (!this.inventoryContext?.timestamp) return true;
            const staleDuration = 5 * 60 * 1000; // 5 minutes
            return (new Date() - this.inventoryContext.timestamp) > staleDuration;
        }
    }

    // Initialize conversation context
    const conversationContext = new ConversationContext();

    // Initialize
    function init() {
        // Auto-resize textarea
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = userInput.scrollHeight + 'px';
        });

        // Handle send button click
        sendButton.addEventListener('click', handleSendMessage);

        // Handle enter key
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });

        // Handle new chat button
        newChatBtn.addEventListener('click', startNewChat);

        // Handle example prompts
        examplePrompts.forEach(prompt => {
            prompt.addEventListener('click', () => {
                startChat(prompt.textContent);
            });
        });
    }

    // Start new chat
    function startNewChat() {
        welcomeScreen.style.display = 'flex';
        chatMessages.style.display = 'none';
        resetPipeline();
        messagesContainer.innerHTML = '';
        
        // Generate new session ID
        chatHistoryManager.currentSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Create new session in database
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: '',
                context: {
                    sessionId: chatHistoryManager.currentSessionId,
                    newSession: true
                }
            })
        }).catch(error => console.error('Error creating new session:', error));
    }

    // Start chat with a message
    function startChat(message = '') {
        welcomeScreen.style.display = 'none';
        chatMessages.style.display = 'block';
        if (message) {
            userInput.value = message;
            handleSendMessage();
        }
    }

    // Handle send message
    async function handleSendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Clear input immediately
        userInput.value = '';
        userInput.style.height = 'auto';

        // Show chat interface
        welcomeScreen.style.display = 'none';
        chatMessages.style.display = 'block';

        // Reset pipeline for new query
        resetPipelineVisualization();

        // Add user message to chat
        appendMessage('user', message);

        try {
            // Start with Query Received
            updatePipelineStatus('query_received', 'in_progress');
            await sleep(500); // Slightly longer delay for visibility
            updatePipelineStatus('query_received', 'completed');

            // Query Enhancement Stage
            updatePipelineStatus('query_enhanced', 'in_progress');
            await sleep(500);

            // Send to backend with session context
            const response = await fetch('/api/ui/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: message,
                    session_id: chatHistoryManager.currentSessionId,
                    include_pipeline_status: true // Request pipeline status updates
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to get response from server');
            }

            const data = await response.json();
            
            // Update Query Enhancement status based on server response
            updatePipelineStatus('query_enhanced', data.pipeline_status?.query_enhanced || 'completed');

            // Router Agent Stage
            updatePipelineStatus('router_agent', 'in_progress');
            await sleep(500);
            updatePipelineStatus('router_agent', data.pipeline_status?.router_agent || 'completed');

            // Parallel Processing of Agent Stages
            const activeAgents = [];
            
            // RAG Agent
            if (data.pipeline_status?.rag_agent !== 'skipped') {
                activeAgents.push({
                    id: 'rag_agent',
                    status: data.pipeline_status?.rag_agent || 'in_progress'
                });
            }

            // Web Search Agent
            if (data.pipeline_status?.web_search_agent !== 'skipped') {
                activeAgents.push({
                    id: 'web_search_agent',
                    status: data.pipeline_status?.web_search_agent || 'in_progress'
                });
            }

            // Analytics Agent
            if (data.pipeline_status?.analytics_agent !== 'skipped') {
                activeAgents.push({
                    id: 'analytics_agent',
                    status: data.pipeline_status?.analytics_agent || 'in_progress'
                });
            }

            // Update active agents in parallel
            if (activeAgents.length > 0) {
                // Show agents starting in parallel
                activeAgents.forEach(agent => {
                    updatePipelineStatus(agent.id, 'in_progress');
                });
                await sleep(500);

                // Update their final statuses
                activeAgents.forEach(agent => {
                    updatePipelineStatus(agent.id, agent.status || 'completed');
                });
                await sleep(300);
            } else {
                // If no agents were activated, mark them as skipped
                updatePipelineStatus('rag_agent', 'skipped');
                updatePipelineStatus('web_search_agent', 'skipped');
                updatePipelineStatus('analytics_agent', 'skipped');
            }

            // Aggregation Stage
            updatePipelineStatus('aggregation_agent', 'in_progress');
            await sleep(500);
            updatePipelineStatus('aggregation_agent', data.pipeline_status?.aggregation_agent || 'completed');

            // Handle response
            if (data.error) {
                throw new Error(data.error);
            }

            // Display response
            appendMessage('assistant', data.content);

            // Update chat history
            await updateChatHistory();

        } catch (error) {
            console.error('Error:', error);
            // Update pipeline to show error state
            markPipelineError();
            appendMessage('assistant', 'Sorry, there was an error processing your request.');
            showErrorMessage(error.message);
        }
    }

    function resetPipelineVisualization() {
        const stages = [
            'query_received',
            'query_enhanced',
            'router_agent',
            'rag_agent',
            'web_search_agent',
            'analytics_agent',
            'aggregation_agent'
        ];
        
        stages.forEach(stage => {
            updatePipelineStatus(stage, 'waiting');
        });
        
        // Reset progress bar
        const progressBar = document.querySelector('.progress-bar');
        const progressText = document.querySelector('.progress-text');
        if (progressBar && progressText) {
            progressBar.style.width = '0%';
            progressText.textContent = '0%';
        }
    }

    function markPipelineError() {
        const stages = [
            'query_received',
            'query_enhanced',
            'router_agent',
            'rag_agent',
            'web_search_agent',
            'analytics_agent',
            'aggregation_agent'
        ];
        
        stages.forEach((stage, index) => {
            if (document.querySelector(`[data-stage="${stage}"] .stage-icon.in_progress`)) {
                // If stage was in progress, mark as error
                updatePipelineStatus(stage, 'error');
            } else if (document.querySelector(`[data-stage="${stage}"] .stage-icon.waiting`)) {
                // If stage was waiting, mark as cancelled
                updatePipelineStatus(stage, 'cancelled');
            }
            // Completed stages remain completed
        });
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function updatePipelineStatus(stage, status) {
        const pipeline = document.querySelector('.pipeline-visualization');
        if (!pipeline) return;

        const stageElement = pipeline.querySelector(`[data-stage="${stage}"]`);
        if (!stageElement) return;

        // Update stage status
        const statusIndicator = stageElement.querySelector('.status-indicator');
        const statusText = stageElement.querySelector('.status-text');
        const icon = stageElement.querySelector('.stage-icon');

        // Remove all status classes
        icon.classList.remove('waiting', 'in_progress', 'completed', 'error', 'cancelled');
        statusIndicator.classList.remove('waiting', 'in_progress', 'completed', 'error', 'cancelled');

        // Add new status class
        icon.classList.add(status);
        statusIndicator.classList.add(status);
        statusText.textContent = getStatusText(status);

        // Add pulse animation for in_progress
        if (status === 'in_progress') {
            icon.classList.add('pulse');
        } else {
            icon.classList.remove('pulse');
        }

        // Update progress bar
        updateProgressBar();
    }

    function getStatusText(status) {
        const statusMap = {
            'waiting': 'Waiting',
            'in_progress': 'In Progress',
            'completed': 'Completed',
            'error': 'Error',
            'cancelled': 'Cancelled'
        };
        return statusMap[status] || 'Unknown';
    }

    function updateProgressBar() {
        const pipeline = document.querySelector('.pipeline-visualization');
        if (!pipeline) return;

        const stages = pipeline.querySelectorAll('.pipeline-stage');
        const completedStages = pipeline.querySelectorAll('.stage-icon.completed').length;
        const progress = Math.round((completedStages / stages.length) * 100);

        const progressBar = document.querySelector('.progress-bar');
        const progressText = document.querySelector('.progress-text');
        if (progressBar && progressText) {
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${progress}%`;
        }
    }

    function analyzeMessageType(message, context) {
        // Check if this is a follow-up visualization request
        if (isVisualizationRelated(message) && context.inventoryContext) {
            return 'contextual_visualization';
        }
        
        // Check if this is a follow-up inventory query
        if (isInventoryRelated(message) && context.inventoryContext && !context.isContextStale()) {
            return 'contextual_inventory';
        }
        
        // Check for ultra-casual messages
        if (isUltraCasual(message)) {
            return 'casual';
        }
        
        // Default checks
        if (isVisualizationRelated(message)) return 'visualization';
        if (isInventoryRelated(message)) return 'inventory';
        return 'business';
    }

    function updateContextFromResponse(results) {
        // Update query type
        conversationContext.lastQueryType = results.type;
        
        // Update inventory context if present
        if (results.inventory_data) {
            conversationContext.updateInventoryContext(results.inventory_data);
        }
        
        // Update visualization context if present
        if (results.visualization_data) {
            conversationContext.updateVisualizationContext({
                data: results.visualization_data,
                type: results.chart_type,
                options: results.chart_options
            });
        }
        
        // Add response to context
        conversationContext.addMessage('assistant', results.response, {
            type: results.type,
            metadata: {
                visualization: results.visualization_data,
                inventory: results.inventory_data
            }
        });
    }

    async function visualizePipelineForResponse(results, messageType) {
        switch (messageType) {
            case 'casual':
                await visualizeUltraFastPipeline();
                    break;
            case 'contextual_visualization':
            case 'visualization':
                await visualizeAnalyticsPipeline();
                    break;
            case 'contextual_inventory':
            case 'inventory':
                await visualizeInventoryPipeline();
                    break;
                default:
                await visualizeFullPipeline();
        }
    }

    function handleResponseWithContext(results) {
        if (results.type === 'visualization' || 
            (results.visualization_data && results.chart_type)) {
            appendFormattedInventoryResponse(
                'assistant',
                results.response,
                results.visualization_data,
                results.chart_type,
                results.chart_options
            );
        } else {
            appendMessage('assistant', results.response);
        }
    }

    // Check if message is ultra-casual (client-side check)
    function isUltraCasual(message) {
        // Expanded casual keywords for comprehensive detection
        const casualKeywords = [
            // Basic greetings
            'hi', 'hello', 'hey', 'howdy', 'greetings', 'yo', 'hiya', 'sup', 'heya', 
            'hi there', 'hello there', 'hey there', 'hola', 'bonjour', 'ciao', 'namaste',
            
            // Polite expressions
            'thanks', 'thank you', 'appreciate it', 'grateful', 'thx', 'ty',
            'please', 'sorry', 'excuse me', 'pardon', 'apologies', 'my bad',
            'no problem', 'np', 'you\'re welcome', 'welcome', 'glad to help',
            
            // Farewells
            'bye', 'goodbye', 'see you', 'farewell', 'take care', 'later', 'cya',
            'see ya', 'catch you later', 'until next time', 'have a good one',
            'have a nice day', 'have a great day', 'talk to you later', 'ttyl',
            
            // Time-based greetings
            'good morning', 'good afternoon', 'good evening', 'good night', 'good day',
            'morning', 'afternoon', 'evening', 'night', 'day', 'happy day',
            
            // Casual questions
            'how are you', 'what\'s up', 'how is it going', 'how are things', 'how goes it',
            'nice to meet you', 'pleasure', 'how do you do', 'whats happening', 'wassup',
            'how have you been', 'how ya doing', 'how you doing', 'hows life', 'how was your day',
            
            // Testing phrases
            'test', 'testing', 'just testing', 'this is a test', 'checking',
            'hello world', 'ping', 'are you there', 'are you working'
        ];
        
        const cleanMessage = message.toLowerCase().trim().replace(/[?!.,]/g, '');
        
        // Check for exact matches
        if (casualKeywords.includes(cleanMessage)) {
            return true;
        }
        
        // Check for first word matches
        const firstWord = cleanMessage.split(' ')[0];
        if (casualKeywords.includes(firstWord)) {
            return true;
        }
        
        // Check for first two words
        if (cleanMessage.split(' ').length >= 2) {
            const firstTwoWords = cleanMessage.split(' ').slice(0, 2).join(' ');
            if (casualKeywords.includes(firstTwoWords)) {
                return true;
            }
        }
        
        // Check for casual patterns
        const casualPatterns = [
            /^(hi|hey|hello|howdy|greetings|yo|hiya|sup|heya)/i,  // Greetings
            /^(good|nice) (morning|afternoon|evening|day|night)/i, // Time-based greetings
            /^how (are|is|do|have|was|were)/i,                    // How are you type questions
            /^(can|could|would|will|may|might) you/i,             // Request patterns
            /^(what|who|where|when|why|how) (are|is|do|can|would)/i, // Information questions
            /^(thanks|thank|thx|ty|appreciate)/i,                 // Gratitude expressions
            /^(sorry|apologies|excuse|pardon)/i,                  // Apology expressions
            /^(good|nice|great|awesome|cool|excellent)/i,         // Positive expressions
            /^(i'm|im) (new|confused|lost|happy|sad|angry|excited)/i, // State expressions
            /^(help|assist|guide|explain|show)( me)?/i,           // Help requests
            /^(test|testing|checking|ping)/i,                     // Testing expressions
            /^(are|is) (you|this|it) (there|working|online|available)/i, // Status questions
            /^(let's|lets) (talk|chat|discuss)/i,                 // Conversation starters
            /^(i|we) (want|need|would like|have) (to|a)/i,        // Need expressions
            /^(just|quick|one) (checking|question|sec|moment)/i,  // Brief interaction markers
            /^(ok|okay|alright|sure|got it|understood)/i,         // Acknowledgments
            /^(bye|goodbye|see you|farewell|later|cya)/i          // Farewells
        ];
        
        for (const pattern of casualPatterns) {
            if (pattern.test(cleanMessage)) {
                return true;
            }
        }
        
        // Check for casual keywords anywhere in the message
        return casualKeywords.some(keyword => cleanMessage.includes(keyword));
    }

    // Check if message is inventory related
    function isInventoryRelated(message) {
        const inventoryKeywords = ['inventory', 'stock', 'supply', 'level', 'current', 'update'];
        const cleanMessage = message.toLowerCase();
        
        return inventoryKeywords.some(keyword => cleanMessage.includes(keyword));
    }

    // Check if message is visualization related
    function isVisualizationRelated(message) {
        const visualKeywords = ['visual', 'chart', 'graph', 'plot', 'diagram', 'dashboard'];
        const cleanMessage = message.toLowerCase();
        
        return visualKeywords.some(keyword => cleanMessage.includes(keyword));
    }

    // Pipeline Visualization Functions
    function resetPipeline() {
        const pipelineVisualization = document.querySelector('.pipeline-visualization');
        pipelineVisualization.removeAttribute('data-active-stage');
        
        Object.values(pipelineStages).forEach(stage => {
            stage.setAttribute('data-status', 'waiting');
            stage.querySelector('.stage-status').textContent = 'Waiting';
        });
    }

    function updatePipeline(stage, status) {
        const pipelineStage = pipelineStages[stage];
        if (!pipelineStage) return;

        // Update stage status
        pipelineStage.setAttribute('data-status', status);
        const statusText = status.charAt(0).toUpperCase() + status.slice(1);
        pipelineStage.querySelector('.stage-status').textContent = statusText;

        // Update pipeline line color
        const pipelineVisualization = document.querySelector('.pipeline-visualization');
        if (status === 'active') {
            // Clear any previous active stages to prevent visual conflicts
            Object.values(pipelineStages).forEach(s => {
                if (s !== pipelineStage && s.getAttribute('data-status') === 'active') {
                    s.setAttribute('data-status', 'completed');
                    s.querySelector('.stage-status').textContent = 'Completed';
                }
            });
            
            // Set the new active stage
            pipelineVisualization.setAttribute('data-active-stage', stage);
        }
    }

    // Ultra-fast pipeline visualization (bypasses router)
    async function visualizeUltraFastPipeline() {
        resetPipeline();
        
        // Simulate ultra-fast pipeline
        updatePipeline('query', 'active');
        await sleep(300);
        updatePipeline('query', 'completed');
        
        // Skip router completely
        updatePipeline('router', 'skipped');
        await sleep(100);
        
        // Skip other stages
        updatePipeline('knowledge', 'skipped');
        updatePipeline('web', 'skipped');
        updatePipeline('analytics', 'skipped');
        await sleep(100);
        
        // Go directly to response generation
        updatePipeline('generator', 'active');
        await sleep(300);
        updatePipeline('generator', 'completed');
        
        updatePipeline('ui', 'active');
        await sleep(300);
        updatePipeline('ui', 'completed');
        
        // Clear active stage at the end
        const pipelineVisualization = document.querySelector('.pipeline-visualization');
            setTimeout(() => {
            pipelineVisualization.removeAttribute('data-active-stage');
        }, 500);
    }

    // Append message to chat
    function appendMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        contentDiv.textContent = content;

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);
    }

    // Append formatted inventory response
    function appendFormattedInventoryResponse(role, content, visualizationData = null, chartType = null, chartOptions = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        
        // Format the inventory response with line breaks
        const formattedContent = content.replace(/\n/g, '<br>');
        contentDiv.innerHTML = formattedContent;

        // Add visualization if available
        if (visualizationData) {
            const visualizationContainer = document.createElement('div');
            visualizationContainer.className = 'visualization-container';
            
            // Add title if available
            if (chartOptions?.title) {
                const titleElement = document.createElement('h3');
                titleElement.textContent = chartOptions.title;
                visualizationContainer.appendChild(titleElement);
            }
            
            // Add chart container
            const chartContainer = document.createElement('div');
            chartContainer.className = 'chart-container';
            
            const canvas = document.createElement('canvas');
            canvas.id = `chart-${Date.now()}`; // Unique ID
            chartContainer.appendChild(canvas);
            
            visualizationContainer.appendChild(chartContainer);
            contentDiv.appendChild(visualizationContainer);
            
            // Initialize chart after DOM is updated
            setTimeout(() => {
                initializeChart(canvas.id, visualizationData, chartType, chartOptions);
            }, 100);
            
            // Update visualization context
            conversationContext.updateVisualizationContext({
                data: visualizationData,
                type: chartType,
                options: chartOptions
            });
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);
    }

    // Initialize chart with dynamic data
    function initializeChart(canvasId, chartData, chartType = 'bar', chartOptions = {}) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        // Default options
        const options = {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: chartOptions.y_axis_label || 'Value'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: chartOptions.x_axis_label || 'Category'
                    }
                }
            },
            plugins: {
                legend: {
                    display: chartType === 'pie' || chartType === 'radar',
                },
                title: {
                    display: !!chartOptions.title,
                    text: chartOptions.title || '',
                    font: {
                        size: 16
                    }
                }
            }
        };
        
        // Special options for different chart types
        if (chartType === 'pie') {
            delete options.scales;
        }
        
        // Create the chart
        new Chart(ctx, {
            type: chartType,
            data: chartData,
            options: options
        });
    }

    // Update recent chats
    async function updateChatHistory() {
        try {
            const response = await fetch('/api/chat-history');
            const data = await response.json();
            const chatHistory = document.getElementById('chat-history');
            chatHistory.innerHTML = '';

            if (!data.sessions || data.sessions.length === 0) {
                chatHistory.innerHTML = '<div class="no-history">No chat history yet</div>';
                return;
            }

            // Sort sessions by timestamp descending (newest first)
            const sortedSessions = data.sessions.sort((a, b) => 
                new Date(b.timestamp) - new Date(a.timestamp)
            );

            for (const session of sortedSessions) {
                const sessionDiv = document.createElement('div');
                sessionDiv.className = 'chat-session';
                sessionDiv.id = `session-${session.id}`;

                // Add session header with timestamp and delete button
                const sessionHeader = document.createElement('div');
                sessionHeader.className = 'session-header';
                const sessionTime = new Date(session.timestamp).toLocaleString();
                sessionHeader.innerHTML = `
                    <span class="session-time">${sessionTime}</span>
                    <button class="delete-session-btn" onclick="deleteSession('${session.id}')">
                        <i class="fas fa-trash"></i>
                    </button>
                `;
                sessionDiv.appendChild(sessionHeader);

                // Sort messages by timestamp ascending (oldest first)
                const sortedMessages = session.messages.sort((a, b) => 
                    new Date(a.timestamp) - new Date(b.timestamp)
                );

                // Add messages
                const messagesDiv = document.createElement('div');
                messagesDiv.className = 'session-messages';
                
                for (const message of sortedMessages) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `chat-message ${message.role}`;
                    messageDiv.id = `message-${message.id}`;
                    
                    const messageContent = document.createElement('div');
                    messageContent.className = 'message-content';
                    messageContent.textContent = message.content;
                    
                    const messageTime = document.createElement('div');
                    messageTime.className = 'message-time';
                    messageTime.textContent = new Date(message.timestamp).toLocaleTimeString();
                    
                    const deleteButton = document.createElement('button');
                    deleteButton.className = 'delete-message-btn';
                    deleteButton.innerHTML = '<i class="fas fa-times"></i>';
                    deleteButton.onclick = () => deleteMessage(message.id);
                    
                    messageDiv.appendChild(messageContent);
                    messageDiv.appendChild(messageTime);
                    messageDiv.appendChild(deleteButton);
                    messagesDiv.appendChild(messageDiv);
                }
                
                sessionDiv.appendChild(messagesDiv);
                chatHistory.appendChild(sessionDiv);
            }
        } catch (error) {
            console.error('Error updating chat history:', error);
            showToast('Error loading chat history', 'error');
        }
    }

    // Helper function to escape HTML
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Format timestamp to relative time
    function formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        // Less than a minute
        if (diff < 60000) {
            return 'Just now';
        }
        // Less than an hour
        if (diff < 3600000) {
            const minutes = Math.floor(diff / 60000);
            return `${minutes}m ago`;
        }
        // Less than a day
        if (diff < 86400000) {
            const hours = Math.floor(diff / 3600000);
            return `${hours}h ago`;
        }
        // Less than a week
        if (diff < 604800000) {
            const days = Math.floor(diff / 86400000);
            return `${days}d ago`;
        }
        // Otherwise return date
        return date.toLocaleDateString();
    }

    // Delete single chat message
    async function deleteChat(messageId) {
        try {
            const response = await fetch(`/api/chat-history/message/${messageId}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error('Failed to delete message');
            }
            
            // Update the chat history display
            await updateChatHistory();
            
            // Show success feedback
            showToast('Message deleted successfully');
        } catch (error) {
            console.error('Error deleting message:', error);
            showToast('Error deleting message', 'error');
        }
    }

    // Delete entire session
    async function deleteSession(sessionId) {
        if (confirm('Are you sure you want to delete this chat session?')) {
            try {
                const response = await fetch(`/api/chat-history/session/${sessionId}`, {
                    method: 'DELETE'
                });
                
                if (!response.ok) {
                    throw new Error('Failed to delete session');
                }
                
                // Update the chat history display
                await updateChatHistory();
                
                // Show success feedback
                showToast('Session deleted successfully');
            } catch (error) {
                console.error('Error deleting session:', error);
                showToast('Error deleting session', 'error');
            }
        }
    }

    // Clear all history
    async function clearAllHistory() {
        if (confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
            try {
                const response = await fetch('/api/chat-history', {
                    method: 'DELETE'
                });
                
                if (!response.ok) {
                    throw new Error('Failed to clear history');
                }
                
                // Update the chat history display
                await updateChatHistory();
                
                // Show success feedback
                showToast('Chat history cleared successfully');
            } catch (error) {
                console.error('Error clearing history:', error);
                showToast('Error clearing chat history', 'error');
            }
        }
    }

    // Show toast notification
    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }, 100);
    }

    // Initialize the app
    init();

    // Add this function at the beginning of the file to help with debugging
    function safelyParseResponse(response) {
        console.log("Raw response received:", response);
        
        // Handle different response formats gracefully
        try {
            // If the response is already an object, return it
            if (typeof response === 'object' && response !== null) {
                console.log("Response is already an object");
                return response;
            }
            
            // Try to parse JSON string
            const parsed = JSON.parse(response);
            console.log("Successfully parsed JSON response:", parsed);
            return parsed;
        } catch (e) {
            console.error("Error parsing response:", e);
            // If parsing fails, wrap the raw response in an object
            return {
                response: String(response),
                type: "text",
                error: "Failed to parse response"
            };
        }
    }

    // Find the function that handles the query submission and replace it with this version
    function submitQuery() {
        const queryInput = document.getElementById('query-input');
        const query = queryInput.value.trim();
        
        if (!query) return;
        
        // Clear input
        queryInput.value = '';
        
        // Get current session ID if available
        const sessionId = localStorage.getItem('currentSessionId');
        
        // Add user message to chat
        addMessageToChat('user', query);
        
        // Show loading indicator
        const loadingMessage = addMessageToChat('assistant', '<div class="loading-dots"><div></div><div></div><div></div></div>');
        
        // Get context if available (including session ID)
        const context = sessionId ? { sessionId } : {};
        
        console.log("Sending query:", query, "with context:", context);
        
        // Send query to API
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                context
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Raw response data:", data);
            
            // Remove loading indicator
            loadingMessage.remove();
            
            // Extract results object, handling both formats
            const results = data.results || data;
            console.log("Processed results:", results);
            
            // Process different response formats
            try {
                // Store the session ID if provided
                if (results.session_id) {
                    localStorage.setItem('currentSessionId', results.session_id);
                }
                
                // Handle visualization responses
                if (results.type === 'visualization') {
                    // Add text response
                    addMessageToChat('assistant', results.response);
                    
                    // Create visualization container
                    const vizContainer = document.createElement('div');
                    vizContainer.className = 'visualization-container';
                    vizContainer.id = `viz-${Date.now()}`;
                    
                    // Add to chat
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'chat-message assistant-message';
                    messageDiv.appendChild(vizContainer);
                    document.getElementById('chat-messages').appendChild(messageDiv);
                    
                    // Create visualization
                    createVisualization(
                        vizContainer.id,
                        results.chart_type || 'bar',
                        results.visualization_data || {},
                        results.chart_options || {}
                    );
                } else {
                    // Regular text response
                    addMessageToChat('assistant', results.response || "Sorry, I couldn't process your request.");
                }
                
                // Update chat history
                updateChatHistory();
            } catch (e) {
                console.error("Error processing response:", e, "Results:", results);
                
                // Try to display the response even if there was an error in processing
                if (typeof results === 'object' && results.response) {
                    addMessageToChat('assistant', results.response);
                } else if (typeof results === 'string') {
                    addMessageToChat('assistant', results);
                } else {
                    addMessageToChat('assistant', "Sorry, there was an error processing your request.");
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loadingMessage.remove();
            
            // Provide a friendly error message
            addMessageToChat('assistant', "Sorry, there was an error processing your request. Please try again.");
            
            // Show toast notification
            showToast("Connection error. Please check your network.");
        });
    }

    function addMessageToChat(role, content) {
        console.log(`Adding ${role} message:`, content);
        
        // Create message element
        const messageEl = document.createElement('div');
        messageEl.className = `chat-message ${role}-message`;
        
        // Create avatar
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        
        if (role === 'user') {
            avatar.innerHTML = '<i class="fas fa-user"></i>';
        } else {
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
        }
        
        // Create message content
        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';
        
        // Handle HTML content (like loading indicators)
        if (content && typeof content === 'string' && content.includes('<div class="loading-dots">')) {
            contentEl.innerHTML = content;
        } else {
            // Format text with line breaks for readability
            const formattedContent = typeof content === 'string' 
                ? content.replace(/\n/g, '<br>') 
                : 'Error: Invalid response';
            contentEl.innerHTML = formattedContent;
        }
        
        // Add elements to message
        messageEl.appendChild(avatar);
        messageEl.appendChild(contentEl);
        
        // Add to chat
        document.getElementById('messages').appendChild(messageEl);
        
        // Scroll to bottom
        const messagesContainer = document.getElementById('messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        return messageEl;
    }
}); 