/**
 * app.js - Main entry point for the RAG System UI
 * 
 * This file handles user interactions, query submission, and displaying results
 * for the Retrieval Augmented Generation (RAG) system interface.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize pipeline visualization
    window.initializePipeline();

    // DOM Elements
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const queryTypeSelect = document.getElementById('query-type');
    const submitButton = document.getElementById('submit-query');
    const responseContainer = document.getElementById('response-container');
    const sourcesContainer = document.getElementById('sources-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    // Event Listeners
    queryForm.addEventListener('submit', handleQuerySubmission);
    
    /**
     * Handle query submission
     * @param {Event} event - Form submission event
     */
    async function handleQuerySubmission(event) {
        event.preventDefault();
        
        const query = queryInput.value.trim();
        const queryType = queryTypeSelect.value;
        
        if (!query) {
            showError('Please enter a query.');
            return;
        }
        
        // Display loading state
        setLoadingState(true);
        window.resetPipeline();
        window.simulatePipelineActivity();
        
        try {
            const response = await submitQuery(query, queryType);
            displayResults(response);
        } catch (error) {
            showError(`Error: ${error.message || 'Failed to process query'}`);
        } finally {
            setLoadingState(false);
        }
    }
    
    /**
     * Submit query to backend API
     * @param {string} query - User query
     * @param {string} queryType - Type of query (factual, analytical, hybrid)
     * @returns {Promise<Object>} Response data
     */
    async function submitQuery(query, queryType) {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                query_type: queryType
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to process query');
        }
        
        return response.json();
    }
    
    /**
     * Display query results in the UI
     * @param {Object} data - Response data from the API
     */
    function displayResults(data) {
        // Clear previous results
        responseContainer.innerHTML = '';
        sourcesContainer.innerHTML = '';
        
        // Display the generated response
        const responseElement = document.createElement('div');
        responseElement.className = 'response-text';
        responseElement.innerHTML = formatResponse(data.response);
        responseContainer.appendChild(responseElement);
        
        // Display sources if available
        if (data.sources && data.sources.length > 0) {
            const sourcesTitle = document.createElement('h3');
            sourcesTitle.textContent = 'Sources';
            sourcesContainer.appendChild(sourcesTitle);
            
            const sourcesList = document.createElement('ul');
            sourcesList.className = 'sources-list';
            
            data.sources.forEach(source => {
                const sourceItem = document.createElement('li');
                sourceItem.className = 'source-item';
                
                const sourceTitle = document.createElement('div');
                sourceTitle.className = 'source-title';
                sourceTitle.textContent = source.title || 'Unnamed Source';
                
                const sourceText = document.createElement('div');
                sourceText.className = 'source-text';
                sourceText.textContent = source.text || '';
                
                const sourceMetadata = document.createElement('div');
                sourceMetadata.className = 'source-metadata';
                sourceMetadata.textContent = `Relevance: ${(source.relevance * 100).toFixed(1)}%`;
                
                sourceItem.appendChild(sourceTitle);
                sourceItem.appendChild(sourceText);
                sourceItem.appendChild(sourceMetadata);
                sourcesList.appendChild(sourceItem);
            });
            
            sourcesContainer.appendChild(sourcesList);
        }
    }
    
    /**
     * Format the response with proper styling
     * @param {string} text - Response text
     * @returns {string} Formatted HTML
     */
    function formatResponse(text) {
        // Convert markdown-like syntax to HTML
        return text
            .replace(/\n\n/g, '</p><p>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');
    }
    
    /**
     * Show error message to the user
     * @param {string} message - Error message
     */
    function showError(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = message;
        
        responseContainer.innerHTML = '';
        responseContainer.appendChild(errorElement);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            errorElement.classList.add('fade-out');
            setTimeout(() => {
                if (responseContainer.contains(errorElement)) {
                    responseContainer.removeChild(errorElement);
                }
            }, 500);
        }, 5000);
    }
    
    /**
     * Set loading state for the UI
     * @param {boolean} isLoading - Whether the application is loading
     */
    function setLoadingState(isLoading) {
        if (isLoading) {
            loadingIndicator.style.display = 'block';
            submitButton.disabled = true;
            submitButton.innerHTML = '<span class="spinner"></span> Processing...';
        } else {
            loadingIndicator.style.display = 'none';
            submitButton.disabled = false;
            submitButton.textContent = 'Submit Query';
        }
    }
}); 