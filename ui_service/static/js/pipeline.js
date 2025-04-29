/**
 * PharmAI Assistant - Pipeline Visualization
 * Handles the visualization of the RAG system processing pipeline
 */

// Pipeline stages configuration
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

// Simulate pipeline activity for demonstration or testing
function simulatePipelineActivity() {
    resetPipeline();
    
    // Simulate each stage with delays
    setTimeout(() => updatePipelineStage('query', 'active'), 500);
    setTimeout(() => updatePipelineStage('query', 'completed'), 1000);
    
    setTimeout(() => updatePipelineStage('router', 'active'), 1500);
    setTimeout(() => updatePipelineStage('router', 'completed'), 2500);
    
    setTimeout(() => updatePipelineStage('knowledge', 'active'), 3000);
    setTimeout(() => updatePipelineStage('knowledge', 'completed'), 4500);
    
    setTimeout(() => updatePipelineStage('web', 'active'), 5000);
    setTimeout(() => updatePipelineStage('web', 'completed'), 6000);
    
    setTimeout(() => updatePipelineStage('analytics', 'active'), 6500);
    setTimeout(() => updatePipelineStage('analytics', 'completed'), 8000);
    
    setTimeout(() => updatePipelineStage('generator', 'active'), 8500);
    setTimeout(() => updatePipelineStage('generator', 'completed'), 10000);
    
    setTimeout(() => updatePipelineStage('ui', 'active'), 10500);
    setTimeout(() => updatePipelineStage('ui', 'completed'), 11000);
}

// Export functions
window.initializePipeline = initializePipeline;
window.updatePipelineStage = updatePipelineStage;
window.resetPipeline = resetPipeline;
window.simulatePipelineActivity = simulatePipelineActivity; 