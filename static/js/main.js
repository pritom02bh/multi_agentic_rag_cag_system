function sendMessage() {
    const userInput = document.getElementById('user-input').value.trim();
    if (userInput === '') return;

    // Add user message to chat
    addMessage('user', userInput);
    
    // Clear input field
    document.getElementById('user-input').value = '';

    // Show loading indicator
    const loadingMessage = addMessage('assistant', '<div class="loading-dots"><div></div><div></div><div></div></div>');
    
    // Log the request being sent
    console.log('Sending request to /api/ui/chat with query:', userInput);

    // Send request to backend
    fetch('/api/ui/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: userInput,
            session_id: sessionId
        })
    })
    .then(response => {
        console.log('Received response with status:', response.status);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Received data:', data);
        
        // Remove loading message
        loadingMessage.remove();
        
        // Add assistant response
        if (data.response) {
            addMessage('assistant', data.response);
            
            // If there's a visualization, add it
            if (data.visualization) {
                addVisualization(data.visualization);
            }
        } else if (data.error) {
            addMessage('assistant', `Error: ${data.error}`);
        } else {
            addMessage('assistant', 'I received your message but encountered an issue processing it. Please try again.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        
        // Remove loading message
        loadingMessage.remove();
        
        // Add error message
        addMessage('assistant', 'Sorry, there was an error processing your request. Please try again later.');
    });
} 