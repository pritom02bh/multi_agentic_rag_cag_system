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
        
        // Check for error responses
        if (data.status === 'error') {
            console.error("Error from API:", data.message || "Unknown error");
            
            // Update pipeline to show error
            updatePipelineStage('router', 'error');
            
            // Add error message to UI
            const errorMsg = data.message || 'Sorry, there was an error processing your request.';
            addMessageToUI('assistant', errorMsg);
            
            // Save error to chat history
            saveMessageToHistory('assistant', errorMsg);
            
            return;
        }
        
        // Process successful response
        const responseMessage = data.response.text;
        const visualizations = data.response.visualizations || [];
        
        // Update pipeline stages
        updatePipelineStage('router', 'completed');
        updatePipelineStage('knowledge', 'completed');
        updatePipelineStage('analytics', 'completed');
        updatePipelineStage('generator', 'completed');
        updatePipelineStage('ui', 'completed');
        
        // Add response text to UI
        addMessageToUI('assistant', responseMessage);
        
        // Handle visualizations if present
        if (visualizations.length > 0) {
            visualizations.forEach(chart => {
                addVisualizationToUI(chart);
            });
        }
        
        // Save response to chat history
        saveMessageToHistory('assistant', responseMessage);
        
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
        if (errorMessage) {
            errorMessage.style.display = 'block';
            errorMessage.textContent = 'Error processing your request. Please try again.';
            
            // Hide error message after 5 seconds
            setTimeout(() => {
                errorMessage.style.display = 'none';
            }, 5000);
        }
        
        // Update pipeline to show error
        updatePipelineStage('query', 'error');
        
        // Add error message to UI
        addMessageToUI('assistant', 'Sorry, there was an error processing your request. Please try again or rephrase your question.');
        
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
    
    // Format content (handle newlines and trim any extra spaces)
    let formattedContent = content.trim().replace(/\n/g, '<br>');
    
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

// Add a visualization to the UI
function addVisualizationToUI(chartData) {
    console.log('Adding visualization to UI:', chartData);
    
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) {
        console.error("Messages container element not found");
        return;
    }
    
    // Create container for the chart
    const chartContainer = document.createElement('div');
    chartContainer.className = 'message assistant-message chart-container';
    
    // Create canvas for the chart
    const canvas = document.createElement('canvas');
    chartContainer.appendChild(canvas);
    messagesContainer.appendChild(chartContainer);
    
    // Create and render the chart using Chart.js
    try {
        new Chart(canvas, chartData);
        scrollToBottom();
    } catch (error) {
        console.error('Error creating chart:', error);
    }
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
    
    // Parse content if it's a JSON string
    let parsedContent;
    if (typeof content === 'string') {
        try {
            parsedContent = JSON.parse(content);
        } catch (e) {
            parsedContent = content;
        }
    } else {
        parsedContent = content;
    }
    
    // Handle structured content
    if (typeof parsedContent === 'object' && parsedContent !== null) {
        // Extract text content
        const textContent = parsedContent.text || parsedContent.formatted_response || parsedContent;
        
        // Format text with proper line breaks and structure
        const formattedText = typeof textContent === 'string' 
            ? textContent.replace(/\\n/g, '<br>').replace(/\n/g, '<br>')
            : 'Error: Invalid response';
        
        contentEl.innerHTML = formattedText;
        
        // Add insights if available
        if (parsedContent.insights) {
            const insightsEl = document.createElement('div');
            insightsEl.className = 'message-insights';
            insightsEl.innerHTML = parsedContent.insights.replace(/\n/g, '<br>');
            contentEl.appendChild(insightsEl);
        }
        
        // Add visualizations if available
        if (parsedContent.visualizations && parsedContent.visualizations.length > 0) {
            parsedContent.visualizations.forEach(chart => {
                const chartContainer = document.createElement('div');
                chartContainer.className = 'chart-container';
                
                // Create canvas for the chart
                const canvas = document.createElement('canvas');
                chartContainer.appendChild(canvas);
                contentEl.appendChild(chartContainer);
                
                // Initialize chart
                try {
                    new Chart(canvas, {
                        type: chart.type,
                        data: chart.data,
                        options: chart.options || {}
                    });
                } catch (error) {
                    console.error('Error creating chart:', error);
                }
            });
        }
    } else {
        // Handle plain text content
        contentEl.innerHTML = typeof parsedContent === 'string' 
            ? parsedContent.replace(/\\n/g, '<br>').replace(/\n/g, '<br>')
            : 'Error: Invalid response';
    }
    
    // Add elements to message
    messageEl.appendChild(avatar);
    messageEl.appendChild(contentEl);
    
    // Add to chat
    const messagesContainer = document.getElementById('messages');
    messagesContainer.appendChild(messageEl);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageEl;
} 