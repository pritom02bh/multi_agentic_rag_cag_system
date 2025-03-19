// Pipeline visualization component
class PipelineVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.stages = [
            {
                id: 'query_received',
                name: 'Query Received',
                description: 'Initial query processing',
                icon: 'fas fa-question-circle',
                status: 'waiting'
            },
            {
                id: 'query_enhanced',
                name: 'Query Enhanced',
                description: 'Adding context and visualization suggestions',
                icon: 'fas fa-magic',
                status: 'waiting'
            },
            {
                id: 'router_agent',
                name: 'Router Agent',
                description: 'Determining optimal agent combination',
                icon: 'fas fa-route',
                status: 'waiting'
            },
            {
                id: 'rag_agent',
                name: 'RAG Agent',
                description: 'Retrieving and processing relevant documents',
                icon: 'fas fa-database',
                status: 'waiting'
            },
            {
                id: 'web_search_agent',
                name: 'Web Search Agent',
                description: 'Searching real-time web information',
                icon: 'fas fa-globe',
                status: 'waiting'
            },
            {
                id: 'analytics_agent',
                name: 'Analytics Agent',
                description: 'Processing data analytics and visualizations',
                icon: 'fas fa-chart-bar',
                status: 'waiting'
            },
            {
                id: 'aggregation_agent',
                name: 'Aggregation Agent',
                description: 'Combining responses and formatting output',
                icon: 'fas fa-layer-group',
                status: 'waiting'
            }
        ];
        this.init();
    }

    init() {
        this.render();
        this.setupEventListeners();
    }

    render() {
        // Clear container
        this.container.innerHTML = '';
        
        // Create pipeline container
        const pipelineContainer = document.createElement('div');
        pipelineContainer.className = 'pipeline-container';
        
        // Create progress bar
        const progressBar = document.createElement('div');
        progressBar.className = 'pipeline-progress';
        progressBar.innerHTML = `
            <div class="progress-bar" role="progressbar" style="width: 0%">
                <span class="progress-text">0%</span>
            </div>
        `;
        
        // Create stages container
        const stagesContainer = document.createElement('div');
        stagesContainer.className = 'pipeline-stages';
        
        // Render each stage
        this.stages.forEach((stage, index) => {
            const stageElement = this.createStageElement(stage, index);
            stagesContainer.appendChild(stageElement);
        });
        
        // Append elements
        pipelineContainer.appendChild(progressBar);
        pipelineContainer.appendChild(stagesContainer);
        this.container.appendChild(pipelineContainer);
    }

    createStageElement(stage, index) {
        const stageElement = document.createElement('div');
        stageElement.className = 'pipeline-stage';
        stageElement.dataset.stage = stage.id;
        
        // Add connecting line except for last stage
        if (index < this.stages.length - 1) {
            stageElement.className += ' with-connector';
        }
        
        stageElement.innerHTML = `
            <div class="stage-icon ${stage.status}">
                <i class="${stage.icon}"></i>
            </div>
            <div class="stage-content">
                <h4>${stage.name}</h4>
                <p>${stage.description}</p>
                <div class="stage-status">
                    <span class="status-indicator ${stage.status}"></span>
                    <span class="status-text">${this.getStatusText(stage.status)}</span>
                </div>
            </div>
        `;
        
        return stageElement;
    }

    getStatusText(status) {
        const statusMap = {
            'waiting': 'Waiting',
            'in_progress': 'In Progress',
            'completed': 'Completed',
            'error': 'Error',
            'cancelled': 'Cancelled'
        };
        return statusMap[status] || 'Unknown';
    }

    updateStage(stageId, status, details = null) {
        const stage = this.stages.find(s => s.id === stageId);
        if (!stage) return;

        stage.status = status;
        if (details) {
            stage.details = details;
        }

        // Update DOM
        const stageElement = this.container.querySelector(`[data-stage="${stageId}"]`);
        if (stageElement) {
            const iconElement = stageElement.querySelector('.stage-icon');
            const statusIndicator = stageElement.querySelector('.status-indicator');
            const statusText = stageElement.querySelector('.status-text');

            // Update classes
            iconElement.className = `stage-icon ${status}`;
            statusIndicator.className = `status-indicator ${status}`;
            statusText.textContent = this.getStatusText(status);

            // Add pulse animation for in_progress
            if (status === 'in_progress') {
                iconElement.classList.add('pulse');
            } else {
                iconElement.classList.remove('pulse');
            }
        }

        this.updateProgress();
    }

    updateProgress() {
        const totalStages = this.stages.length;
        const completedStages = this.stages.filter(s => s.status === 'completed').length;
        const progress = Math.round((completedStages / totalStages) * 100);

        const progressBar = this.container.querySelector('.progress-bar');
        const progressText = this.container.querySelector('.progress-text');

        progressBar.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;
    }

    setupEventListeners() {
        // Add event listeners for stage interactions if needed
        this.stages.forEach(stage => {
            const stageElement = this.container.querySelector(`[data-stage="${stage.id}"]`);
            if (stageElement) {
                stageElement.addEventListener('click', () => {
                    this.onStageClick(stage);
                });
            }
        });
    }

    onStageClick(stage) {
        // Show stage details or handle click events
        console.log(`Stage clicked: ${stage.name}`);
        if (stage.details) {
            // Show details in a modal or tooltip
            this.showStageDetails(stage);
        }
    }

    showStageDetails(stage) {
        // Implementation for showing stage details
        // This could be a modal, tooltip, or side panel
        console.log('Stage details:', stage.details);
    }

    reset() {
        this.stages.forEach(stage => {
            stage.status = 'waiting';
            delete stage.details;
        });
        this.render();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PipelineVisualization;
} 