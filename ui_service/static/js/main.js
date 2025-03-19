/**
 * PharmAI Assistant - Main JavaScript
 * Enhanced UI functionality for the pharmaceutical supply chain assistant
 */

// Global variables
let sessionId = null;
let systemOnline = true;
let activeChat = false;
let chatHistory = [];
let typingTimeout = null;

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded");
    
    // Cache DOM elements
    const welcomeScreen = document.getElementById('welcome-screen');
    const chatMessages = document.getElementById('chat-messages');
    const messagesContainer = document.getElementById('messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const newChatButton = document.getElementById('new-chat-btn');
    const recentChats = document.getElementById('recent-chats');
    const loadingIndicator = document.getElementById('chat-loading');
    const errorMessage = document.getElementById('chat-error');
    const pipelineVisualization = document.getElementById('pipeline-visualization');
    
    // Debug element existence
    console.log("Send button exists:", !!sendButton);
    console.log("Chat input exists:", !!chatInput);
    console.log("Pipeline visualization exists:", !!pipelineVisualization);
    
    // Create confirmation modal
    createConfirmationModal();
    
    // Ensure messages container is properly set up
    if (messagesContainer) {
        messagesContainer.style.display = 'flex';
        messagesContainer.style.flexDirection = 'column';
        messagesContainer.style.width = '100%';
    }
    
    // Set up event listeners
    if (sendButton) {
        sendButton.addEventListener('click', function() {
            console.log("Send button clicked");
            sendMessage();
        });
    }
    
    if (chatInput) {
        // Send message on Enter key (but allow Shift+Enter for new line)
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log("Enter key pressed");
                sendMessage();
            }
        });
        
        // Auto-resize textarea as user types
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = (chatInput.scrollHeight) + 'px';
        });
    }
    
    if (newChatButton) {
        // Start new chat
        newChatButton.addEventListener('click', startNewChat);
    }
    
    // Handle window resize
    window.addEventListener('resize', () => {
        if (activeChat) {
            scrollToBottom();
        }
    });
    
    // Check system status
    checkSystemStatus();
    setInterval(checkSystemStatus, 60000); // Check every minute
    
    // Load chat history from local storage
    loadChatHistory();
    
    // Initialize pipeline visualization
    initializePipeline();
    
    // Populate example queries
    populateExampleQueries();
});

// Create confirmation modal
function createConfirmationModal() {
    const modal = document.createElement('div');
    modal.className = 'confirm-modal';
    modal.id = 'confirm-modal';
    
    modal.innerHTML = `
        <div class="confirm-modal-content">
            <div class="confirm-modal-title">Confirm Deletion</div>
            <div class="confirm-modal-text" id="confirm-modal-text">Are you sure you want to delete this chat?</div>
            <div class="confirm-modal-buttons">
                <button class="confirm-modal-button confirm-cancel" id="confirm-cancel">Cancel</button>
                <button class="confirm-modal-button confirm-delete" id="confirm-delete">Delete</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Add event listeners
    document.getElementById('confirm-cancel').addEventListener('click', () => {
        closeConfirmModal();
    });
    
    // Close modal when clicking outside
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeConfirmModal();
        }
    });
}

// Show confirmation modal
function showConfirmModal(message, callback) {
    const modal = document.getElementById('confirm-modal');
    const modalText = document.getElementById('confirm-modal-text');
    const confirmButton = document.getElementById('confirm-delete');
    
    modalText.textContent = message;
    modal.classList.add('active');
    
    // Remove previous event listener
    const newConfirmButton = confirmButton.cloneNode(true);
    confirmButton.parentNode.replaceChild(newConfirmButton, confirmButton);
    
    // Add new event listener
    newConfirmButton.addEventListener('click', () => {
        callback();
        closeConfirmModal();
    });
}

// Close confirmation modal
function closeConfirmModal() {
    const modal = document.getElementById('confirm-modal');
    modal.classList.remove('active');
}

// Check system status
function checkSystemStatus() {
    fetch(API_CONFIG.statusEndpoint)
        .then(response => response.json())
        .then(data => {
            systemOnline = data.status === 'success';
            updateSystemStatus(systemOnline);
        })
        .catch(error => {
            console.error('Error checking system status:', error);
            systemOnline = false;
            updateSystemStatus(false);
        });
}

// Update system status indicators
function updateSystemStatus(isOnline) {
    const statusIndicator = document.querySelector('.status-indicator');
    if (statusIndicator) {
        statusIndicator.className = `status-indicator ${isOnline ? 'online' : 'offline'}`;
        statusIndicator.title = isOnline ? 'System Online' : 'System Offline';
    }
}

// Initialize pipeline visualization
function initializePipeline() {
    console.log("Initializing pipeline visualization");
    const pipelineVisualization = document.getElementById('pipeline-visualization');
    if (!pipelineVisualization) {
        console.error("Pipeline visualization element not found");
        return;
    }
    
    // Clear existing pipeline
    pipelineVisualization.innerHTML = '';
    
    // Create pipeline stages
    const stages = [
        { id: 'query', name: 'Query Received', icon: 'fa-question' },
        { id: 'router', name: 'Router Agent', icon: 'fa-random' },
        { id: 'knowledge', name: 'Knowledge Base', icon: 'fa-database' },
        { id: 'web', name: 'Web Search', icon: 'fa-globe' },
        { id: 'analytics', name: 'Analytics', icon: 'fa-chart-bar' },
        { id: 'generator', name: 'Response Generator', icon: 'fa-cogs' },
        { id: 'ui', name: 'UI Response', icon: 'fa-reply' }
    ];
    
    // Add stages to pipeline
    stages.forEach((stage, index) => {
        const stageElement = document.createElement('div');
        stageElement.className = 'pipeline-stage waiting';
        stageElement.id = `stage-${stage.id}`;
        
        stageElement.innerHTML = `
            <div class="stage-icon">
                <i class="fas ${stage.icon}"></i>
            </div>
            <div class="stage-info">
                <div class="stage-name">${stage.name}</div>
                <div class="stage-status">Waiting</div>
            </div>
        `;
        pipelineVisualization.appendChild(stageElement);
    });
    
    console.log("Pipeline visualization initialized with", stages.length, "stages");
}

// Update pipeline stage
function updatePipelineStage(stageId, status) {
    console.log(`Updating pipeline stage: ${stageId} to ${status}`);
    const stage = document.getElementById(`stage-${stageId}`);
    if (!stage) {
        console.error(`Stage element not found: stage-${stageId}`);
        return;
    }
    
    // Remove existing status classes
    stage.classList.remove('waiting', 'active', 'completed', 'error');
    
    // Add new status class
    stage.classList.add(status);
    
    // Update status text
    const statusElement = stage.querySelector('.stage-status');
    if (statusElement) {
        statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }
    
    // Add processing class to pipeline visualization when any stage is active
    const pipelineVisualization = document.getElementById('pipeline-visualization');
    if (pipelineVisualization) {
        if (status === 'active') {
            pipelineVisualization.classList.add('processing');
        } else {
            // Check if any stage is still active
            const activeStages = document.querySelectorAll('.pipeline-stage.active');
            if (activeStages.length === 0) {
                pipelineVisualization.classList.remove('processing');
            }
        }
    }
}

// Reset pipeline to initial state
function resetPipeline() {
    console.log("Resetting pipeline");
    const stages = ['query', 'router', 'knowledge', 'web', 'analytics', 'generator', 'ui'];
    stages.forEach(stage => {
        updatePipelineStage(stage, 'waiting');
    });
    
    // Remove processing class
    const pipelineVisualization = document.getElementById('pipeline-visualization');
    if (pipelineVisualization) {
        pipelineVisualization.classList.remove('processing');
    }
}

// Populate example queries
function populateExampleQueries() {
    const promptGrid = document.querySelector('.prompt-grid');
    if (!promptGrid) {
        console.error("Prompt grid element not found");
        return;
    }
    
    const exampleQueries = [
        "What is the current inventory status?",
        "Are there any medications with low stock?",
        "Show me the supply chain for Paracetamol",
        "What are the recent shipment delays?",
        "Analyze our inventory turnover rate",
        "What are the best practices for pharmaceutical storage?"
    ];
    
    promptGrid.innerHTML = '';
    
    exampleQueries.forEach(query => {
        const promptItem = document.createElement('div');
        promptItem.className = 'prompt-item';
        promptItem.textContent = query;
        promptItem.addEventListener('click', () => {
            const chatInput = document.getElementById('chat-input');
            if (chatInput) {
                chatInput.value = query;
                chatInput.dispatchEvent(new Event('input'));
                startNewChat();
                setTimeout(() => sendMessage(), 100);
            }
        });
        promptGrid.appendChild(promptItem);
    });
    
    console.log("Example queries populated");
}

// Load chat history from local storage
function loadChatHistory() {
    try {
        const savedHistory = localStorage.getItem('chatHistory');
        if (savedHistory) {
            chatHistory = JSON.parse(savedHistory);
            updateChatHistorySidebar();
            console.log("Chat history loaded:", chatHistory.length, "chats");
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
        chatHistory = [];
    }
}

// Save chat history to local storage
function saveChatHistory() {
    try {
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    } catch (error) {
        console.error('Error saving chat history:', error);
    }
}

// Update chat history sidebar
function updateChatHistorySidebar() {
    const historyContainer = document.querySelector('.chat-history');
    if (!historyContainer) {
        console.error("History container element not found");
        return;
    }
    
    // Update or create history header
    let historyHeader = historyContainer.querySelector('.history-header');
    if (!historyHeader) {
        historyHeader = document.createElement('div');
        historyHeader.className = 'history-header';
        
        const historyLabel = document.createElement('div');
        historyLabel.className = 'history-label';
        historyLabel.textContent = 'RECENT';
        
        const deleteAllBtn = document.createElement('button');
        deleteAllBtn.className = 'delete-all-btn';
        deleteAllBtn.innerHTML = '<i class="fas fa-trash-alt"></i> Clear All';
        deleteAllBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (chatHistory.length > 0) {
                showConfirmModal('Are you sure you want to delete all chats?', deleteAllChats);
            }
        });
        
        historyHeader.appendChild(historyLabel);
        historyHeader.appendChild(deleteAllBtn);
        
        // Insert at the beginning
        if (historyContainer.firstChild) {
            historyContainer.insertBefore(historyHeader, historyContainer.firstChild);
        } else {
            historyContainer.appendChild(historyHeader);
        }
    }
    
    const recentChats = document.getElementById('recent-chats');
    if (!recentChats) {
        console.error("Recent chats element not found");
        return;
    }
    
    recentChats.innerHTML = '';
    
    // Display most recent chats first (up to 10)
    const recentChatHistory = [...chatHistory].reverse();
    
    recentChatHistory.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.className = 'chat-item';
        if (chat.id === sessionId) {
            chatItem.classList.add('active');
        }
        
        // Get first message or use default title
        const title = chat.messages && chat.messages.length > 0 
            ? chat.messages[0].content.substring(0, 25) + (chat.messages[0].content.length > 25 ? '...' : '')
            : 'New Chat';
        
        chatItem.innerHTML = `
            <div class="chat-item-header">
                <div class="chat-item-title">${title}</div>
                <button class="delete-chat-btn" title="Delete chat">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="chat-item-time">${formatDate(chat.timestamp)}</div>
        `;
        
        // Add event listener for loading chat
        chatItem.addEventListener('click', () => {
            loadChat(chat.id);
        });
        
        // Add event listener for delete button
        const deleteBtn = chatItem.querySelector('.delete-chat-btn');
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent chat from loading when delete is clicked
            showConfirmModal('Are you sure you want to delete this chat?', () => deleteChat(chat.id));
        });
        
        recentChats.appendChild(chatItem);
    });
}

// Format date for chat history
function formatDate(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    
    // If today, show time
    if (date.toDateString() === now.toDateString()) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // If this year, show month and day
    if (date.getFullYear() === now.getFullYear()) {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
    
    // Otherwise show full date
    return date.toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' });
}

// Start a new chat
function startNewChat() {
    console.log("Starting new chat");
    
    // Generate new session ID
    sessionId = generateSessionId();
    
    // Create new chat in history
    const newChat = {
        id: sessionId,
        timestamp: new Date().toISOString(),
        messages: []
    };
    
    chatHistory.push(newChat);
    saveChatHistory();
    updateChatHistorySidebar();
    
    // Clear messages
    const messagesContainer = document.getElementById('messages');
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
    }
    
    // Show chat interface
    const welcomeScreen = document.getElementById('welcome-screen');
    const chatMessages = document.getElementById('chat-messages');
    
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    if (chatMessages) chatMessages.style.display = 'block';
    
    // Reset pipeline
    resetPipeline();
    
    // Set active chat flag
    activeChat = true;
    
    // Focus input
    const chatInput = document.getElementById('chat-input');
    if (chatInput) chatInput.focus();
}

// Load an existing chat
function loadChat(chatId) {
    console.log("Loading chat:", chatId);
    
    // Find chat in history
    const chat = chatHistory.find(c => c.id === chatId);
    if (!chat) {
        console.error("Chat not found:", chatId);
        return;
    }
    
    // Set session ID
    sessionId = chat.id;
    
    // Update sidebar
    updateChatHistorySidebar();
    
    // Clear messages
    const messagesContainer = document.getElementById('messages');
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
    }
    
    // Add messages
    if (chat.messages && chat.messages.length > 0) {
        chat.messages.forEach(message => {
            addMessageToUI(message.role, message.content);
        });
    }
    
    // Show chat interface
    const welcomeScreen = document.getElementById('welcome-screen');
    const chatMessages = document.getElementById('chat-messages');
    
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    if (chatMessages) chatMessages.style.display = 'block';
    
    // Reset pipeline
    resetPipeline();
    
    // Set active chat flag
    activeChat = true;
    
    // Scroll to bottom
    scrollToBottom();
    
    // Focus input
    const chatInput = document.getElementById('chat-input');
    if (chatInput) chatInput.focus();
}

// Generate a session ID
function generateSessionId() {
    return 'session_' + Math.random().toString(36).substring(2, 15);
}

// Send a message
function sendMessage() {
    console.log("sendMessage function called");
    
    const chatInput = document.getElementById('chat-input');
    if (!chatInput) {
        console.error("Chat input element not found");
        return;
    }
    
    const message = chatInput.value.trim();
    if (!message) {
        console.log("Message is empty, not sending");
        return;
    }
    
    console.log("Sending message:", message);
    
    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // If no active chat, start one
    if (!activeChat) {
        startNewChat();
    }
    
    // Add message to UI
    addMessageToUI('user', message);
    
    // Save message to chat history
    saveMessageToHistory('user', message);
    
    // Show loading indicator
    const loadingIndicator = document.getElementById('chat-loading');
    const errorMessage = document.getElementById('chat-error');
    
    if (loadingIndicator) loadingIndicator.style.display = 'block';
    if (errorMessage) errorMessage.style.display = 'none';
    
    // Update pipeline - query received
    updatePipelineStage('query', 'completed');
    updatePipelineStage('router', 'active');
    
    console.log("Sending message to API:", message);
    
    // Send message to API
    fetch(API_CONFIG.chatEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: message,
            session_id: sessionId
        })
    })
    .then(response => {
        console.log("API response status:", response.status);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("API response data:", data);
        
        // Hide loading indicator
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        
        // Update pipeline based on the pipeline_status in response
        if (data.pipeline_status) {
            for (const [stage, status] of Object.entries(data.pipeline_status)) {
                if (stage === 'query_received' && status === 'completed') {
                    updatePipelineStage('query', 'completed');
                } else if (stage === 'router_agent' && status === 'completed') {
                    updatePipelineStage('router', 'completed');
                } else if (stage === 'knowledge_base' && status === 'completed') {
                    updatePipelineStage('knowledge', 'completed');
                } else if (stage === 'web_search' && status === 'completed') {
                    updatePipelineStage('web', 'completed');
                } else if (stage === 'policy_transport' && status === 'completed') {
                    updatePipelineStage('analytics', 'completed');
                } else if (stage === 'response_generation' && status === 'completed') {
                    updatePipelineStage('generator', 'completed');
                } else if (stage === 'ui_response' && status === 'completed') {
                    updatePipelineStage('ui', 'completed');
                }
            }
        } else {
            // Fallback for older response format
            if (data.type === 'casual') {
                // Simple response path
                updatePipelineStage('router', 'completed');
                updatePipelineStage('generator', 'completed');
                updatePipelineStage('ui', 'completed');
            } else if (data.type === 'business') {
                // Business query path
                updatePipelineStage('router', 'completed');
                updatePipelineStage('knowledge', 'completed');
                updatePipelineStage('analytics', 'completed');
                updatePipelineStage('generator', 'completed');
                updatePipelineStage('ui', 'completed');
            } else if (data.type === 'complex') {
                // Complex query path
                updatePipelineStage('router', 'completed');
                updatePipelineStage('knowledge', 'completed');
                updatePipelineStage('web', 'completed');
                updatePipelineStage('analytics', 'completed');
                updatePipelineStage('generator', 'completed');
                updatePipelineStage('ui', 'completed');
            } else if (data.type === 'error') {
                // Error path
                updatePipelineStage('router', 'error');
            }
        }
        
        // Extract the message content from the response
        let responseContent = '';
        if (data.message) {
            responseContent = data.message;
        } else if (data.content) {
            responseContent = data.content;
        } else if (data.response) {
            responseContent = data.response;
        } else if (data.error) {
            responseContent = `Error: ${data.error}`;
            updatePipelineStage('router', 'error');
        } else {
            responseContent = 'Sorry, I received your message but encountered an issue processing it. Please try again.';
        }
        
        // Add response to UI
        addMessageToUI('assistant', responseContent);
        
        // Save response to chat history
        saveMessageToHistory('assistant', responseContent);
        
        // Update session ID if provided
        if (data.session_id) {
            sessionId = data.session_id;
        }
    })
    .catch(error => {
        console.error('Error sending message:', error);
        
        // Hide loading indicator
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        
        // Show error message
        if (errorMessage) errorMessage.style.display = 'block';
        
        // Update pipeline to show error
        updatePipelineStage('query', 'error');
        
        // Add error message to UI
        addMessageToUI('assistant', 'Sorry, there was an error processing your request.');
        
        // Save error to chat history
        saveMessageToHistory('assistant', 'Sorry, there was an error processing your request.');
    });
}

// Scroll to bottom of messages - improved version
function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    
    if (chatMessages) {
        // Use requestAnimationFrame for smoother scrolling
        requestAnimationFrame(() => {
            // Force scroll to absolute bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
            console.log("Scrolled to bottom, height:", chatMessages.scrollHeight);
        });
    }
}

// Add a message to the UI
function addMessageToUI(role, content) {
    console.log(`Adding ${role} message to UI:`, content.substring(0, 50) + (content.length > 50 ? '...' : ''));
    
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) {
        console.error("Messages container element not found");
        return;
    }
    
    // Ensure messages container exists and is properly configured
    if (!document.getElementById('messages')) {
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            const newMessagesContainer = document.createElement('div');
            newMessagesContainer.id = 'messages';
            newMessagesContainer.className = 'messages-container';
            chatMessages.appendChild(newMessagesContainer);
            messagesContainer = newMessagesContainer;
        }
    }
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role}-message`;
    
    // Format content 
    // If the content is very large (inventory listings, etc.), format it with better styling
    let formattedContent = content.trim();
    
    // Check for rate limit errors first (these need special handling)
    if (role === 'assistant' && formattedContent.includes('Error code: 429') && formattedContent.includes('rate_limit_exceeded')) {
        formattedContent = formatErrorMessage(formattedContent);
    }
    // Use appropriate formatting based on the length and structure of the content
    else if (role === 'assistant' && formattedContent.length > 500 && (formattedContent.includes('Item ID:') || formattedContent.includes('inventory'))) {
        // This looks like a detailed inventory response, use enhanced formatting
        formattedContent = formatInventoryContent(formattedContent);
    } else {
        // Standard formatting with just line breaks
        formattedContent = formattedContent.replace(/\n/g, '<br>');
    }
    
    // Create message HTML with minimal whitespace
    let messageHTML = '';
    if (role === 'user') {
        messageHTML = `<div class="message-avatar"><i class="fas fa-user"></i></div><div class="message-content"><div class="message-text">${formattedContent}</div></div>`;
    } else {
        messageHTML = `<div class="message-avatar"><i class="fas fa-robot"></i></div><div class="message-content"><div class="message-text">${formattedContent}</div></div>`;
    }
    
    messageElement.innerHTML = messageHTML;
    messagesContainer.appendChild(messageElement);
    
    // Force immediate scroll to bottom
    scrollToBottom();
    
    // Add event listener to ensure scrolling works when images load
    const images = messageElement.querySelectorAll('img');
    if (images.length > 0) {
        images.forEach(img => {
            img.addEventListener('load', scrollToBottom);
        });
    }
    
    // Add MutationObserver to watch for content changes
    const observer = new MutationObserver(scrollToBottom);
    
    // Start observing the message element for content changes
    observer.observe(messageElement, { 
        childList: true, 
        subtree: true, 
        characterData: true 
    });
    
    // Disconnect after 2 seconds to avoid performance issues
    setTimeout(() => {
        observer.disconnect();
    }, 2000);
}

// Format error messages for better display
function formatErrorMessage(errorContent) {
    // Extract the most important parts of the error
    let errorType = 'Error';
    let errorMessage = errorContent;
    
    // Try to extract API error if available
    if (errorContent.includes('Error code:')) {
        const errorCodeMatch = errorContent.match(/Error code: (\d+)/);
        if (errorCodeMatch && errorCodeMatch[1]) {
            errorType = `API Error ${errorCodeMatch[1]}`;
        }
        
        // Try to extract the actual error message
        if (errorContent.includes('message')) {
            try {
                // Extract just the message part
                const messageMatch = errorContent.match(/'message': '([^']+)'/);
                if (messageMatch && messageMatch[1]) {
                    errorMessage = messageMatch[1];
                }
            } catch (e) {
                console.error("Error parsing error message:", e);
                // Keep the original message as fallback
            }
        }
    }
    
    // Create a styled error message
    return `
        <div class="error-container">
            <div class="error-title">${errorType}</div>
            <div class="error-message">${errorMessage}</div>
            <div class="error-solution">
                <p>This is a rate limit error from the API. Please try one of the following:</p>
                <ul>
                    <li>Wait a few moments and try your query again</li>
                    <li>Break your request into smaller, more specific questions</li>
                    <li>Try a different type of query</li>
                </ul>
            </div>
        </div>
        <style>
            .error-container {
                background-color: rgba(231, 76, 60, 0.1);
                border-left: 4px solid #e74c3c;
                border-radius: 6px;
                padding: 15px;
                margin: 10px 0;
            }
            .error-title {
                color: #c0392b;
                font-weight: bold;
                font-size: 1.1em;
                margin-bottom: 10px;
            }
            .error-message {
                margin-bottom: 15px;
                font-weight: 500;
            }
            .error-solution {
                background-color: rgba(255, 255, 255, 0.5);
                padding: 10px;
                border-radius: 4px;
            }
            .error-solution ul {
                margin-left: 20px;
                margin-top: 5px;
            }
            .error-solution li {
                margin-bottom: 5px;
            }
        </style>
    `;
}

// Format inventory content with better styling
function formatInventoryContent(content) {
    // Split into lines for processing
    const lines = content.split('\n');
    let formattedHTML = '';
    
    // Process each line
    let inItemBlock = false;
    let itemCount = 0;
    
    for (const line of lines) {
        // Check if this is an item header
        if (line.match(/^\d+\.\s+Item ID:/)) {
            // If we were in an item block, close it
            if (inItemBlock) {
                formattedHTML += '</div></div>';
            }
            
            // Start a new item block
            itemCount++;
            inItemBlock = true;
            formattedHTML += `<div class="inventory-item">
                <div class="inventory-item-header">${line}</div>
                <div class="inventory-item-details">`;
        } 
        // Check if this is an item detail
        else if (line.match(/^\s*-?\s*[A-Za-z]/) && inItemBlock) {
            // It's a detail line - remove leading dashes or spaces
            const detailText = line.replace(/^\s*-?\s*/, '');
            
            if (!detailText.trim()) continue; // Skip empty lines
            
            // Apply special formatting based on the type of detail
            if (detailText.startsWith('Current Stock:')) {
                formattedHTML += `<div class="inventory-detail stock">${detailText}</div>`;
            } else if (detailText.startsWith('Expiry Date:')) {
                formattedHTML += `<div class="inventory-detail expiry">${detailText}</div>`;
            } else if (detailText.startsWith('Storage Condition:')) {
                formattedHTML += `<div class="inventory-detail storage">${detailText}</div>`;
            } else if (detailText.startsWith('Lead Time:')) {
                formattedHTML += `<div class="inventory-detail lead-time">${detailText}</div>`;
            } else if (detailText.startsWith('Reorder Point:')) {
                formattedHTML += `<div class="inventory-detail reorder">${detailText}</div>`;
            } else if (detailText.startsWith('Unit Cost:') || detailText.startsWith('Selling Price:')) {
                formattedHTML += `<div class="inventory-detail price">${detailText}</div>`;
            } else if (detailText.startsWith('Batch Number:')) {
                formattedHTML += `<div class="inventory-detail batch">${detailText}</div>`;
            } else if (detailText.startsWith('Manufacturing Date:')) {
                formattedHTML += `<div class="inventory-detail manufacturing">${detailText}</div>`;
            } else if (detailText.startsWith('Special Handling:')) {
                formattedHTML += `<div class="inventory-detail handling">${detailText}</div>`;
            } else {
                formattedHTML += `<div class="inventory-detail">${detailText}</div>`;
            }
        } 
        // If it's just a blank line in an item block, add spacing
        else if (line.trim() === '' && inItemBlock) {
            // Don't add anything for blank lines within item blocks
            continue;
        } 
        // If it's the summary line after all items
        else if (line.includes('comprehensive overview') || line.includes('These details')) {
            // Close any open item block
            if (inItemBlock) {
                formattedHTML += '</div></div>';
                inItemBlock = false;
            }
            formattedHTML += `<div class="inventory-summary">${line}</div>`;
        } 
        // If it's the intro line
        else if (line.includes('Based on the information')) {
            formattedHTML += `<div class="inventory-intro">${line}</div>`;
        } 
        // Otherwise, just add the line as-is
        else {
            if (inItemBlock) {
                // Skip empty lines within item blocks
                if (line.trim()) {
                    formattedHTML += `<div class="inventory-text">${line}</div>`;
                }
            } else {
                // Skip empty lines outside item blocks
                if (line.trim()) {
                    formattedHTML += `${line}<br>`;
                }
            }
        }
    }
    
    // Close any open item block
    if (inItemBlock) {
        formattedHTML += '</div></div>';
    }
    
    // Add CSS for inventory formatting
    formattedHTML = `
        <style>
            .inventory-item {
                margin-bottom: 20px;
                border: 1px solid #2a3042;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .inventory-item-header {
                background-color: #2a3042;
                color: #fff;
                padding: 10px 12px;
                font-weight: bold;
                font-size: 1.1em;
            }
            .inventory-item-details {
                padding: 15px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 8px;
            }
            .inventory-detail {
                padding: 8px 10px;
                border-radius: 6px;
                background-color: rgba(42, 48, 66, 0.05);
                margin-bottom: 5px;
            }
            .inventory-detail.stock {
                background-color: rgba(52, 152, 219, 0.15);
                font-weight: 600;
            }
            .inventory-detail.expiry {
                background-color: rgba(231, 76, 60, 0.15);
                font-weight: 600;
            }
            .inventory-detail.storage {
                background-color: rgba(155, 89, 182, 0.15);
            }
            .inventory-detail.price {
                background-color: rgba(46, 204, 113, 0.15);
            }
            .inventory-detail.reorder {
                background-color: rgba(243, 156, 18, 0.15);
            }
            .inventory-detail.handling {
                background-color: rgba(231, 76, 60, 0.1);
            }
            .inventory-detail.batch {
                background-color: rgba(52, 73, 94, 0.1);
            }
            .inventory-detail.manufacturing {
                background-color: rgba(41, 128, 185, 0.1);
            }
            .inventory-intro, .inventory-summary {
                margin: 15px 0;
                padding: 12px 15px;
                background-color: rgba(42, 48, 66, 0.05);
                border-radius: 8px;
                line-height: 1.5;
            }
            .inventory-intro {
                font-weight: 600;
                border-left: 4px solid #3498db;
            }
            .inventory-summary {
                border-left: 4px solid #2ecc71;
            }
            .inventory-text {
                padding: 5px 0;
            }
        </style>
    ` + formattedHTML;
    
    return formattedHTML;
}

// Save a message to chat history
function saveMessageToHistory(role, content) {
    // Find current chat
    const currentChat = chatHistory.find(chat => chat.id === sessionId);
    if (!currentChat) {
        console.error("Current chat not found in history");
        return;
    }
    
    // Add message
    currentChat.messages.push({
        role,
        content,
        timestamp: new Date().toISOString()
    });
    
    // Update timestamp
    currentChat.timestamp = new Date().toISOString();
    
    // Save to local storage
    saveChatHistory();
    
    // Update sidebar
    updateChatHistorySidebar();
}

// Remove the periodic check and replace with a better scroll handler
const chatMessages = document.getElementById('chat-messages');
if (chatMessages) {
    chatMessages.addEventListener('scroll', () => {
        // Store the scroll position in a data attribute
        chatMessages.dataset.scrollTop = chatMessages.scrollTop;
        chatMessages.dataset.scrollHeight = chatMessages.scrollHeight;
    });
}

// Make functions available globally
window.startNewChat = startNewChat;
window.sendMessage = sendMessage;
window.loadChat = loadChat;

// Delete a chat
function deleteChat(chatId) {
    console.log("Deleting chat:", chatId);
    
    // Remove from chat history
    chatHistory = chatHistory.filter(chat => chat.id !== chatId);
    
    // Save to local storage
    saveChatHistory();
    
    // Update sidebar
    updateChatHistorySidebar();
    
    // If the deleted chat was the active one, start a new chat
    if (chatId === sessionId) {
        startNewChat();
    }
}

// Delete all chats
function deleteAllChats() {
    console.log("Deleting all chats");
    
    // Clear chat history
    chatHistory = [];
    
    // Save to local storage
    saveChatHistory();
    
    // Update sidebar
    updateChatHistorySidebar();
    
    // Start a new chat
    startNewChat();
} 