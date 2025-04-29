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
    const chatInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
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
    
    // Create and display timestamp
    addTimestamp();
    
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
    // Remove existing modal if any
    const existingModal = document.getElementById('confirm-modal');
    if (existingModal) {
        existingModal.remove();
    }

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
}

// Show confirmation modal
function showConfirmModal(message, callback) {
    const modal = document.getElementById('confirm-modal');
    if (!modal) {
        createConfirmationModal();
    }
    
    const modalText = document.getElementById('confirm-modal-text');
    const confirmButton = document.getElementById('confirm-delete');
    const cancelButton = document.getElementById('confirm-cancel');
    
    if (modalText) modalText.textContent = message;
    
    // Remove existing event listeners
    const newConfirmButton = confirmButton.cloneNode(true);
    const newCancelButton = cancelButton.cloneNode(true);
    
    confirmButton.parentNode.replaceChild(newConfirmButton, confirmButton);
    cancelButton.parentNode.replaceChild(newCancelButton, cancelButton);
    
    // Add new event listeners
    newConfirmButton.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        callback();
        closeConfirmModal();
    };
    
    newCancelButton.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        closeConfirmModal();
    };
    
    // Show modal
    modal.style.display = 'flex';
    modal.classList.add('active');
    
    // Close modal when clicking outside
    modal.onclick = (e) => {
        if (e.target === modal) {
            closeConfirmModal();
        }
    };
}

// Close confirmation modal
function closeConfirmModal() {
    const modal = document.getElementById('confirm-modal');
    if (modal) {
        modal.classList.remove('active');
        modal.style.display = 'none';
    }
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

// Define pipeline stages
const PIPELINE_STAGES = [
    { id: 'query', name: 'Query Received', icon: 'fa-question' },
    { id: 'router', name: 'Router Agent', icon: 'fa-random' },
    { id: 'knowledge', name: 'Knowledge Base', icon: 'fa-database' },
    { id: 'web', name: 'Web Search', icon: 'fa-globe' },
    { id: 'analytics', name: 'Analytics', icon: 'fa-chart-bar' },
    { id: 'generator', name: 'Response Generator', icon: 'fa-cogs' },
    { id: 'ui', name: 'UI Response', icon: 'fa-reply' }
];

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
    
    // Add stages to pipeline
    PIPELINE_STAGES.forEach((stage, index) => {
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
    
    console.log("Pipeline visualization initialized with", PIPELINE_STAGES.length, "stages");
}

// Update pipeline stage status
function updatePipelineStage(stageId, status) {
    const stageElement = document.getElementById(`stage-${stageId}`);
    if (!stageElement) {
        console.error(`Pipeline stage element not found: ${stageId}`);
        return;
    }
    
    // Remove current status classes
    stageElement.classList.remove('waiting', 'active', 'completed', 'error');
    
    // Add new status class
    stageElement.classList.add(status);
    
    // Update status text
    const statusElement = stageElement.querySelector('.stage-status');
    if (statusElement) {
        statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }
    
    // If this stage is active, mark all previous stages as completed
    if (status === 'active') {
        const stageIndex = PIPELINE_STAGES.findIndex(stage => stage.id === stageId);
        if (stageIndex > 0) {
            for (let i = 0; i < stageIndex; i++) {
                updatePipelineStage(PIPELINE_STAGES[i].id, 'completed');
            }
        }
    }
    
    console.log(`Updated pipeline stage ${stageId} to ${status}`);
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
            const chatInput = document.getElementById('userInput');
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
            
            // Convert string timestamps to Date objects
            chatHistory.forEach(chat => {
                chat.timestamp = new Date(chat.timestamp).getTime();
            });
            
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
    const recentChats = document.getElementById('recent-chats');
    if (!recentChats) return;

    // Clear existing chats
    recentChats.innerHTML = '';

    // Sort chats by timestamp (most recent first)
    const sortedChats = [...chatHistory].sort((a, b) => b.timestamp - a.timestamp);

    // Add each chat to the sidebar
    sortedChats.forEach(chat => {
        // Get the first user message as the title
        const firstUserMessage = chat.messages?.find(m => m.role === 'user')?.content || 'Untitled Chat';
        
        // Create chat item container
        const chatItem = document.createElement('div');
        chatItem.className = 'chat-item';
        chatItem.dataset.chatId = chat.id;
        if (chat.id === sessionId) {
            chatItem.classList.add('active');
        }

        // Create chat content
        const chatContent = document.createElement('div');
        chatContent.className = 'chat-item-content';

        // Create left section
        const chatLeft = document.createElement('div');
        chatLeft.className = 'chat-item-left';

        // Create and set up title
        const title = document.createElement('div');
        title.className = 'chat-item-title';
        title.textContent = truncateText(firstUserMessage, 40);

        // Create and set up timestamp
        const time = document.createElement('div');
        time.className = 'chat-item-time';
        time.textContent = formatTimestamp(chat.timestamp);

        // Create delete button
        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'delete-chat';
        deleteBtn.innerHTML = '<i class="fas fa-times"></i>';
        deleteBtn.setAttribute('aria-label', 'Delete chat');
        deleteBtn.title = 'Delete chat';

        // Set up delete button click handler
        deleteBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            const chatId = this.closest('.chat-item').dataset.chatId;
            if (chatId) {
                showConfirmModal('Are you sure you want to delete this chat?', () => {
                    deleteChat(chatId);
                });
            }
        });

        // Set up chat item click handler
        chatItem.addEventListener('click', function(e) {
            if (!e.target.closest('.delete-chat')) {
                const chatId = this.dataset.chatId;
                if (chatId) {
                    loadChat(chatId);
                }
            }
        });

        // Assemble the components
        chatLeft.appendChild(title);
        chatLeft.appendChild(time);
        chatContent.appendChild(chatLeft);
        chatItem.appendChild(chatContent);
        chatItem.appendChild(deleteBtn);
        recentChats.appendChild(chatItem);
    });

    // Update empty state
    if (sortedChats.length === 0) {
        const emptyState = document.createElement('div');
        emptyState.className = 'empty-state';
        emptyState.innerHTML = `
            <div class="empty-state-icon">
                <i class="fas fa-comments"></i>
            </div>
            <div class="empty-state-text">No chat history</div>
        `;
        recentChats.appendChild(emptyState);
    }
}

// Format timestamp
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) {
        return date.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
        });
    } else if (hours > 0) {
        return `${hours}h ago`;
    } else if (minutes > 0) {
        return `${minutes}m ago`;
    } else {
        return 'Just now';
    }
}

// Start a new chat
function startNewChat() {
    console.log("Starting new chat");
    
    // Set active chat to false first to reset UI
    activeChat = false;
    
    // Reset the UI (will clear messages and show chat area)
    resetUI();
    
    // Generate a new session ID
    sessionId = generateSessionId();
    
    // Explicitly reset the pipeline to ensure it's in waiting state
    resetPipeline();
    
    // Set active chat to true after resetting UI
    activeChat = true;
    
    // Focus the input field
    const userInput = document.getElementById('userInput');
    if (userInput) {
        userInput.focus();
    }
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
            addMessageToUI(message);
        });
    }
    
    // Show chat interface
    const welcomeScreen = document.getElementById('welcome-screen');
    const chatMessages = document.getElementById('chat-messages');
    
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    if (chatMessages) {
        chatMessages.style.display = 'flex';
        chatMessages.style.flexDirection = 'column';
    }
    
    // Reset pipeline
    resetPipeline();
    
    // Set active chat flag
    activeChat = true;
    
    // Scroll to bottom
    scrollToBottom();
    
    // Focus input
    const chatInput = document.getElementById('userInput');
    if (chatInput) chatInput.focus();
}

// Generate a session ID
function generateSessionId() {
    return 'session_' + Math.random().toString(36).substring(2, 15);
}

// Reset pipeline to initial state
function resetPipeline() {
    console.log("Resetting pipeline");
    // Ensure all pipeline stages are reset to waiting state
    PIPELINE_STAGES.forEach(stage => {
        const stageElement = document.getElementById(`stage-${stage.id}`);
        if (stageElement) {
            // Remove all status classes
            stageElement.classList.remove('active', 'completed', 'error');
            stageElement.classList.add('waiting');
            
            // Reset status text
            const statusElement = stageElement.querySelector('.stage-status');
            if (statusElement) {
                statusElement.textContent = 'Waiting';
            }
        }
    });
}

function sendMessage() {
    const userInput = document.getElementById('userInput');
    if (!userInput) return;
    
    const query = userInput.value.trim();
    if (!query) return;
    
    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';
    
    // If this is a new chat, reset the UI
    if (!activeChat) {
        resetUI();
        activeChat = true;
    } else {
        // For existing chats, ensure pipeline is reset for the new query
        resetPipeline();
    }
    
    // Generate session ID if needed
    if (!sessionId) {
        sessionId = generateSessionId();
    }
    
    // Show chat messages area
    const welcomeScreen = document.getElementById('welcome-screen');
    const chatMessages = document.getElementById('chat-messages');
    
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    if (chatMessages) {
        chatMessages.style.display = 'flex';
        chatMessages.style.flexDirection = 'column';
    }
    
    // Add user message
    addMessageToUI({
        role: 'user',
        content: query
    });
    
    // Update pipeline visualization - start with query received
    updatePipelineStage('query', 'active');
    
    // Show loading state
    setLoading(true);
    
    // Debug
    console.log("Sending query:", query, "with session ID:", sessionId);
    
    // Prepare API request
    const requestData = {
        query: query,
        session_id: sessionId
    };
    
    // Send to API
    fetch(API_CONFIG.chatEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('API response:', data);
        
        // Extract metadata about which components were actually used
        const pipelineMetadata = data.pipeline_metadata || {};
        const activeAgent = pipelineMetadata.active_agent || 'vector_rag';
        const activeSources = pipelineMetadata.active_sources || [];
        
        // Helper function to handle stage animation
        function animateStage(stageId, isActive, nextCallback) {
            if (isActive) {
                updatePipelineStage(stageId, 'active');
                setTimeout(() => {
                    updatePipelineStage(stageId, 'completed');
                    if (nextCallback) nextCallback();
                }, 300);
            } else {
                // Make sure inactive stages show as 'waiting' not 'completed'
                const stageElement = document.getElementById(`stage-${stageId}`);
                if (stageElement) {
                    stageElement.classList.remove('active', 'completed', 'error');
                    stageElement.classList.add('waiting');
                    
                    // Update status text
                    const statusElement = stageElement.querySelector('.stage-status');
                    if (statusElement) {
                        statusElement.textContent = 'Waiting';
                    }
                }
                
                if (nextCallback) nextCallback();
            }
        }
        
        // Mark query stage as completed (this is always used)
        updatePipelineStage('query', 'completed');
        
        // Router stage (always used)
        updatePipelineStage('router', 'active');
        setTimeout(() => {
            updatePipelineStage('router', 'completed');
            
            // Knowledge base is used if any source other than 'web' is in the active sources
            const knowledgeBaseSources = ['inventory', 'transport', 'guidelines', 'policy'];
            const usedKnowledgeBase = activeSources.some(source => knowledgeBaseSources.includes(source));
            
            // Start knowledge base animation
            animateStage('knowledge', usedKnowledgeBase, () => {
                // Web search is only used if 'web' is in the sources
                const usedWebSearch = activeSources.includes('web');
                
                // Start web search animation
                animateStage('web', usedWebSearch, () => {
                    // Analytics is only used for analytics or hybrid agent types
                    const usedAnalytics = activeAgent === 'analytics' || activeAgent === 'hybrid';
                    
                    // Start analytics animation
                    animateStage('analytics', usedAnalytics, () => {
                        // Response generator is always used
                        animateStage('generator', true, () => {
                            // UI response is always used
                            animateStage('ui', true);
                        });
                    });
                });
            });
        }, 300);
        
        // Clear loading state
        setLoading(false);
        
        // If we have a response, add it to the UI
        if (data.response) {
            // Save the chat title if this is the first message
            if (chatHistory.length === 0 || !chatHistory.find(chat => chat.id === sessionId)) {
                // Create a new chat entry
                const newChat = {
                    id: sessionId,
                    title: truncateText(query, 30),
                    timestamp: new Date().toISOString(),
                    messages: []
                };
                
                // Add to history
                chatHistory.push(newChat);
                saveChatHistory();
                updateChatHistorySidebar();
            }
            
            // Add response to UI
            addMessageToUI({
                role: 'assistant',
                content: data.response
            });
            
            // Scroll to bottom
            scrollToBottom();
        } else {
            showError('No response received');
            updatePipelineStage('ui', 'error');
        }
    })
    .catch(error => {
        console.error('Error sending message:', error);
        setLoading(false);
        showError('Error: ' + error.message);
        
        // Mark all stages as error
        PIPELINE_STAGES.forEach(stage => {
            if (stage.id !== 'query') { // Keep query stage as completed
                updatePipelineStage(stage.id, 'error');
            }
        });
    });
    
    // Scroll to bottom
    scrollToBottom();
}

// Add chart to UI
function addChartToUI(chart) {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) {
        console.error("Messages container not found");
        return;
    }
    
    const chartElement = document.createElement('div');
    chartElement.className = 'message assistant-message chart-message';
    
    chartElement.innerHTML = `
        <div class="message-avatar"><i class="fas fa-chart-bar"></i></div>
        <div class="message-content">
            <div class="chart-title">${chart.title || 'Analysis Chart'}</div>
            <div class="chart-container" id="chart-${Date.now()}"></div>
            ${chart.description ? `<div class="chart-description">${chart.description}</div>` : ''}
        </div>
    `;
    
    messagesContainer.appendChild(chartElement);
    
    // Render chart using Chart.js
    const chartContainer = chartElement.querySelector('.chart-container');
    if (chartContainer && chart.data) {
        const ctx = document.createElement('canvas');
        chartContainer.appendChild(ctx);
        
        new Chart(ctx, {
            type: chart.type || 'bar',
            data: chart.data,
            options: chart.options || {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
    
    scrollToBottom();
}

// Scroll to the bottom of the messages container
function scrollToBottom() {
    const messagesContainer = document.getElementById('messages');
    const chatMessages = document.getElementById('chat-messages');
    
    if (messagesContainer && chatMessages) {
        // Use requestAnimationFrame to ensure the DOM has updated
        requestAnimationFrame(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Double-check scroll position after a short delay
            setTimeout(() => {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 100);
        });
    }
}

// Function to add a message to the UI
function addMessageToUI(message) {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) {
        console.error("Messages container not found");
        return;
    }
    
    // Create message element
    const messageElement = document.createElement('div');
    messageElement.className = `message message-${message.role}`;
    
    // Create message icon (avatar)
    const iconElement = document.createElement('div');
    iconElement.className = 'message-icon';
    
    // Set icon based on role - use more modern icons
    if (message.role === 'user') {
        iconElement.innerHTML = '<i class="fas fa-user"></i>';
    } else if (message.role === 'assistant') {
        iconElement.innerHTML = '<i class="fas fa-brain"></i>';
    } else {
        iconElement.innerHTML = '<i class="fas fa-exclamation-circle"></i>';
    }
    
    // Create message content
    const contentElement = document.createElement('div');
    contentElement.className = 'message-content';
    
    // Process message content (convert markdown and sanitize)
    let processedContent = '';
    try {
        // Use marked to parse markdown
        processedContent = marked.parse(message.content);
        // Use DOMPurify to sanitize HTML
        processedContent = DOMPurify.sanitize(processedContent, {
            ALLOWED_TAGS: ['p', 'br', 'ul', 'ol', 'li', 'strong', 'em', 'code', 'pre', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'a', 'hr', 'img'],
            ALLOWED_ATTR: ['href', 'target', 'src', 'alt', 'class']
        });
    } catch (e) {
        console.error("Error processing message content:", e);
        processedContent = `<p>${message.content}</p>`;
    }
    
    // Set the content
    contentElement.innerHTML = processedContent;
    
    // Create message footer with token count (only for assistant messages)
    if (message.role === 'assistant') {
        const footerElement = document.createElement('div');
        footerElement.className = 'message-footer';
        
        // Action buttons
        const actionsElement = document.createElement('div');
        actionsElement.className = 'message-actions';
        
        // Copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'message-action-button';
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.title = 'Copy to clipboard';
        copyButton.addEventListener('click', () => {
            const text = message.content;
            navigator.clipboard.writeText(text).then(() => {
                // Show a brief notification that text was copied
                copyButton.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                }, 2000);
            });
        });
        
        // Other action buttons
        const actionsHTML = `
            <button class="message-action-button" title="Like response"><i class="far fa-thumbs-up"></i></button>
            <button class="message-action-button" title="Dislike response"><i class="far fa-thumbs-down"></i></button>
            <button class="message-action-button" title="More options"><i class="fas fa-ellipsis-h"></i></button>
        `;
        
        actionsElement.appendChild(copyButton);
        actionsElement.innerHTML += actionsHTML;
        
        // Token count (randomize for demo or use actual count if available)
        const tokenCount = message.tokens || Math.floor(Math.random() * 100) + 20;
        const tokenCountElement = document.createElement('div');
        tokenCountElement.className = 'token-count';
        tokenCountElement.textContent = `${tokenCount} tokens`;
        
        // Add elements to footer
        footerElement.appendChild(actionsElement);
        footerElement.appendChild(tokenCountElement);
        
        // Add footer to content
        contentElement.appendChild(footerElement);
    }
    
    // Add message components to the message element
    messageElement.appendChild(iconElement);
    messageElement.appendChild(contentElement);
    
    // Add the message element to the container
    messagesContainer.appendChild(messageElement);
    
    // Scroll to the bottom of the messages container
    scrollToBottom();
    
    // Save to history
    saveMessageToHistory(message);
}

// Function to save message to history
function saveMessageToHistory(message) {
    // Find the current chat in history
    const chat = chatHistory.find(c => c.id === sessionId);
    if (!chat) return;
    
    // Add message to chat
    chat.messages.push(message);
    
    // Save updated history
    saveChatHistory();
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
    
    // Find the chat to delete
    const chatToDelete = chatHistory.find(chat => chat.id === chatId);
    if (!chatToDelete) {
        console.error("Chat not found:", chatId);
        return;
    }
    
    // Remove from chat history
    chatHistory = chatHistory.filter(chat => chat.id !== chatId);
    
    // Save to local storage
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    
    // Update sidebar
    updateChatHistorySidebar();
    
    // If the deleted chat was active, show welcome screen
    if (chatId === sessionId) {
        showWelcomeScreen();
        sessionId = null;
    }
}

// Delete all chats
function deleteAllChats() {
    console.log("Deleting all chats");
    
    // Clear chat history
    chatHistory = [];
    sessionId = null;
    
    // Save to local storage
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    
    // Update sidebar
    updateChatHistorySidebar();
    
    // Show welcome screen
    showWelcomeScreen();
}

// Show welcome screen
function showWelcomeScreen() {
    console.log("Showing welcome screen");
    const welcomeScreen = document.getElementById('welcome-screen');
    const chatMessages = document.getElementById('chat-messages');
    
    if (welcomeScreen && chatMessages) {
        welcomeScreen.style.display = 'flex';
        chatMessages.style.display = 'none';
    }
    
    // Reset active chat flag
    activeChat = false;
    
    // Ensure pipeline is reset when showing welcome screen
    resetPipeline();
}

// Add event listener for Clear All button
document.addEventListener('DOMContentLoaded', () => {
    const clearAllButton = document.getElementById('clear-all-chats');
    if (clearAllButton) {
        clearAllButton.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (chatHistory.length > 0) {
                showConfirmModal('Are you sure you want to delete all chats?', deleteAllChats);
            }
        });
    }
});

// Helper function to truncate text
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength - 1) + '…';
}

// Set loading state
function setLoading(isLoading) {
    const loadingIndicator = document.getElementById('chat-loading');
    if (loadingIndicator) {
        loadingIndicator.style.display = isLoading ? 'block' : 'none';
    }
    
    // Disable/enable input during loading
    const chatInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    
    if (chatInput) chatInput.disabled = isLoading;
    if (sendButton) sendButton.disabled = isLoading;
}

// Show error message
function showError(message) {
    const errorMessage = document.getElementById('chat-error');
    if (errorMessage) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        
        // Hide error after 5 seconds
        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 5000);
    }
    
    // Add error to chat if needed
    addMessageToUI({
        role: 'system',
        content: `Error: ${message}`
    });
}

// Reset UI for new chat
function resetUI() {
    console.log("Resetting UI for new chat");
    
    // Clear messages
    const messagesContainer = document.getElementById('messages');
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
    }
    
    // Show chat messages area
    const welcomeScreen = document.getElementById('welcome-screen');
    const chatMessages = document.getElementById('chat-messages');
    
    if (welcomeScreen) welcomeScreen.style.display = 'none';
    if (chatMessages) {
        chatMessages.style.display = 'flex';
        chatMessages.style.flexDirection = 'column';
    }
    
    // Reset pipeline visualization
    resetPipeline();
    
    // Clear errors
    const errorMessage = document.getElementById('chat-error');
    if (errorMessage) {
        errorMessage.style.display = 'none';
    }
    
    // Reset loading state
    setLoading(false);
}

// Function to add timestamp to the messages
function addTimestamp() {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;
    
    const timestamp = document.createElement('div');
    timestamp.className = 'message-timestamp';
    
    // Format current date
    const now = new Date();
    const formattedTime = now.toLocaleString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
    
    timestamp.textContent = formattedTime;
    messagesContainer.appendChild(timestamp);
}

// Function to show notification
function showNotification(message) {
    // Create notification element if it doesn't exist
    let notification = document.getElementById('notification');
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'notification';
        notification.style.position = 'fixed';
        notification.style.bottom = '20px';
        notification.style.left = '50%';
        notification.style.transform = 'translateX(-50%)';
        notification.style.padding = '10px 16px';
        notification.style.backgroundColor = 'rgba(0, 255, 133, 0.9)';
        notification.style.color = '#000';
        notification.style.borderRadius = '8px';
        notification.style.zIndex = '1000';
        notification.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.2)';
        notification.style.fontWeight = '500';
        notification.style.fontSize = '14px';
        notification.style.opacity = '0';
        notification.style.transition = 'opacity 0.3s ease';
        document.body.appendChild(notification);
    }
    
    // Set message and show notification
    notification.textContent = message;
    notification.style.opacity = '1';
    
    // Hide after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
    }, 3000);
}

function displayResponse(data) {
    const response = data.response;
    const responseContainer = document.querySelector('#response-container');
    const loadingIndicator = document.querySelector('#loading-indicator');
    
    // Hide loading indicator
    loadingIndicator.style.display = 'none';
    
    // Build response HTML
    let responseHTML = `<div class="response-text">${formatResponse(response)}</div>`;
    
    // Add sources if available
    if (data.source_info && data.source_info.trim()) {
        responseHTML += `<div class="sources-section">
            <div class="sources-header">
                <i class="fas fa-info-circle"></i> Sources
            </div>
            <div class="sources-content">
                ${formatSources(data.source_info)}
            </div>
        </div>`;
    }
    
    // Add charts if available
    if (data.charts && data.charts.length > 0) {
        responseHTML += `<div class="charts-section">
            <div class="charts-header">
                <i class="fas fa-chart-bar"></i> Visualizations
            </div>
            <div class="charts-container" id="charts-container">
                ${generateChartContainers(data.charts)}
            </div>
        </div>`;
    }
    
    // Display in container
    responseContainer.innerHTML = responseHTML;
    responseContainer.style.display = 'block';
    
    // Initialize charts if needed
    if (data.charts && data.charts.length > 0) {
        initCharts(data.charts);
    }
    
    // Update pipeline visualization
    updatePipelineVisualization(data);
}

function formatSources(sourcesText) {
    // Format the sources text for HTML display
    if (!sourcesText) return '';
    
    // Replace new lines with <br> tags and add bullet styling
    return sourcesText
        .replace(/•/g, '<span class="source-bullet">•</span>')
        .replace(/\n/g, '<br>');
} 