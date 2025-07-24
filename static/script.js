static// ProphitBet Football Prediction Application - JavaScript

// Global state management
const ProphitBet = {
    currentPage: '',
    isLoading: false,
    alerts: [],
    
    // Initialize the application
    init() {
        this.setupEventListeners();
        this.initializeComponents();
        this.setCurrentPage();
    },
    
    // Set up global event listeners
    setupEventListeners() {
        // Navigation handling
        document.addEventListener('DOMContentLoaded', () => {
            this.init();
        });
        
        // Modal handling
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.closeModal(e.target);
            }
        });
        
        // File upload handling
        document.addEventListener('change', (e) => {
            if (e.target.type === 'file') {
                this.handleFileSelect(e);
            }
        });
        
        // Form submission handling
        document.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleFormSubmit(e);
        });
    },
    
    // Initialize page-specific components
    initializeComponents() {
        this.initializeFileUploads();
        this.initializeCharts();
        this.initializeModals();
        this.initializeButtons();
    },
    
    // Initialize buttons with safe event handling
    initializeButtons() {
        // Safely add event listeners only if elements exist
        const downloadBtn = document.getElementById('download-league-btn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', (e) => {
                const form = e.target.closest('form');
                if (form) {
                    LeagueManager.downloadLeague(new FormData(form));
                }
            });
        }
        
        const trainBtn = document.getElementById('train-model-btn');
        if (trainBtn) {
            trainBtn.addEventListener('click', (e) => {
                const form = e.target.closest('form');
                if (form) {
                    ModelManager.trainModel(new FormData(form));
                }
            });
        }
    },
    
    // Set current page for navigation highlighting
    setCurrentPage() {
        const path = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-item a');
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === path) {
                link.classList.add('active');
            }
        });
        
        // Set current page
        if (path.includes('leagues')) this.currentPage = 'leagues';
        else if (path.includes('models')) this.currentPage = 'models';
        else if (path.includes('predictions')) this.currentPage = 'predictions';
        else if (path.includes('analysis')) this.currentPage = 'analysis';
        else this.currentPage = 'dashboard';
    },
    
    // Show loading spinner
    showLoading(element = null) {
        this.isLoading = true;
        if (element) {
            const spinner = document.createElement('div');
            spinner.className = 'spinner';
            spinner.id = 'loading-spinner';
            element.appendChild(spinner);
        }
    },
    
    // Hide loading spinner
    hideLoading(element = null) {
        this.isLoading = false;
        if (element) {
            const spinner = element.querySelector('#loading-spinner');
            if (spinner) spinner.remove();
        } else {
            const spinners = document.querySelectorAll('#loading-spinner');
            spinners.forEach(spinner => spinner.remove());
        }
    },
    
    // Show alert message
    showAlert(message, type = 'info', duration = 5000) {
        const alertId = 'alert-' + Date.now();
        const alertHtml = `
            <div id="${alertId}" class="alert alert-${type}" role="alert">
                ${message}
                <button type="button" class="close" onclick="ProphitBet.closeAlert('${alertId}')" style="float: right; background: none; border: none; font-size: 1.2rem; cursor: pointer;">×</button>
            </div>
        `;
        
        const alertContainer = document.getElementById('alert-container') || this.createAlertContainer();
        alertContainer.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                this.closeAlert(alertId);
            }, duration);
        }
    },
    
    // Create alert container if it doesn't exist
    createAlertContainer() {
        const container = document.createElement('div');
        container.id = 'alert-container';
        container.style.position = 'fixed';
        container.style.top = '20px';
        container.style.right = '20px';
        container.style.zIndex = '9999';
        container.style.maxWidth = '400px';
        document.body.appendChild(container);
        return container;
    },
    
    // Close alert
    closeAlert(alertId) {
        const alert = document.getElementById(alertId);
        if (alert) {
            alert.style.transition = 'opacity 0.3s ease';
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 300);
        }
    },
    
    // Handle form submissions
    async handleFormSubmit(e) {
        const form = e.target;
        const formData = new FormData(form);
        const action = form.action || window.location.pathname;
        const method = form.method || 'POST';
        
        try {
            this.showLoading(form);
            
            const response = await fetch(action, {
                method: method,
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            this.handleApiResponse(result);
            
        } catch (error) {
            this.showAlert(`Error: ${error.message}`, 'danger');
        } finally {
            this.hideLoading(form);
        }
    },
    
    // Handle API responses
    handleApiResponse(response) {
        if (response.error) {
            this.showAlert(response.error, 'danger');
        } else if (response.message) {
            this.showAlert(response.message, 'success');
        }
        
        // Refresh data if needed
        if (response.refresh) {
            setTimeout(() => window.location.reload(), 1000);
        }
    },
    
    // Initialize file uploads
    initializeFileUploads() {
        const fileUploads = document.querySelectorAll('.file-upload');
        
        fileUploads.forEach(upload => {
            upload.addEventListener('dragover', (e) => {
                e.preventDefault();
                upload.classList.add('dragover');
            });
            
            upload.addEventListener('dragleave', () => {
                upload.classList.remove('dragover');
            });
            
            upload.addEventListener('drop', (e) => {
                e.preventDefault();
                upload.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const fileInput = upload.querySelector('input[type="file"]');
                    if (fileInput) {
                        fileInput.files = files;
                        this.handleFileSelect({ target: fileInput });
                    }
                }
            });
        });
    },
    
    // Handle file selection
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        const fileInfo = document.querySelector('.file-info');
        if (fileInfo) {
            fileInfo.innerHTML = `
                <p><strong>Selected file:</strong> ${file.name}</p>
                <p><strong>Size:</strong> ${this.formatFileSize(file.size)}</p>
                <p><strong>Type:</strong> ${file.type}</p>
            `;
        }
        
        // Validate file type for CSV uploads
        if (e.target.accept && e.target.accept.includes('.csv')) {
            if (!file.name.toLowerCase().endsWith('.csv')) {
                this.showAlert('Please select a CSV file.', 'warning');
                e.target.value = '';
                return;
            }
        }
    },
    
    // Format file size
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    // Initialize charts
    initializeCharts() {
        // Only initialize if Chart.js is available
        if (typeof Chart !== 'undefined') {
            this.setupCharts();
        }
    },
    
    // Setup charts with Chart.js
    setupCharts() {
        const chartElements = document.querySelectorAll('[data-chart]');
        
        chartElements.forEach(element => {
            const chartType = element.dataset.chart;
            const chartData = JSON.parse(element.dataset.chartData || '{}');
            
            new Chart(element, {
                type: chartType,
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        });
    },
    
    // Initialize modals
    initializeModals() {
        const modalTriggers = document.querySelectorAll('[data-modal]');
        
        modalTriggers.forEach(trigger => {
            trigger.addEventListener('click', (e) => {
                e.preventDefault();
                const modalId = trigger.dataset.modal;
                this.openModal(modalId);
            });
        });
        
        const closeButtons = document.querySelectorAll('.modal .close');
        closeButtons.forEach(button => {
            button.addEventListener('click', () => {
                const modal = button.closest('.modal');
                this.closeModal(modal);
            });
        });
    },
    
    // Open modal
    openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden';
        }
    },
    
    // Close modal
    closeModal(modal) {
        if (typeof modal === 'string') {
            modal = document.getElementById(modal);
        }
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    }
};

// League Management Functions
const LeagueManager = {
    // Download league data
    async downloadLeague(leagueName) {
        try {
            ProphitBet.showLoading();
            
            const response = await fetch('/api/leagues/download', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ league_name: leagueName })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                ProphitBet.showAlert(result.message, 'success');
                setTimeout(() => window.location.reload(), 2000);
            } else {
                ProphitBet.showAlert(result.error, 'danger');
            }
            
        } catch (error) {
            ProphitBet.showAlert(`Error downloading league: ${error.message}`, 'danger');
        } finally {
            ProphitBet.hideLoading();
        }
    },
    
    // Delete league data
    async deleteLeague(leagueName) {
        if (!confirm(`Are you sure you want to delete ${leagueName} data?`)) {
            return;
        }
        
        try {
            const response = await fetch('/api/leagues/delete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ league_name: leagueName })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                ProphitBet.showAlert(result.message, 'success');
                setTimeout(() => window.location.reload(), 1000);
            } else {
                ProphitBet.showAlert(result.error, 'danger');
            }
            
        } catch (error) {
            ProphitBet.showAlert(`Error deleting league: ${error.message}`, 'danger');
        }
    }
};

// Model Management Functions
const ModelManager = {
    // Train new model
    async trainModel(formData) {
        try {
            ProphitBet.showLoading();
            
            const response = await fetch('/api/models/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    league_name: formData.get('league_name'),
                    model_type: formData.get('model_type'),
                    model_name: formData.get('model_name')
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                ProphitBet.showAlert(result.message, 'success');
                ProphitBet.closeModal('trainModelModal');
                setTimeout(() => window.location.reload(), 2000);
            } else {
                ProphitBet.showAlert(result.error, 'danger');
            }
            
        } catch (error) {
            ProphitBet.showAlert(`Error training model: ${error.message}`, 'danger');
        } finally {
            ProphitBet.hideLoading();
        }
    },
    
    // Delete model
    async deleteModel(modelName) {
        if (!confirm(`Are you sure you want to delete the model "${modelName}"?`)) {
            return;
        }
        
        try {
            const response = await fetch('/api/models/delete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model_name: modelName })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                ProphitBet.showAlert(result.message, 'success');
                setTimeout(() => window.location.reload(), 1000);
            } else {
                ProphitBet.showAlert(result.error, 'danger');
            }
            
        } catch (error) {
            ProphitBet.showAlert(`Error deleting model: ${error.message}`, 'danger');
        }
    }
};

// Prediction Functions
const PredictionManager = {
    // Make single prediction
    async predictSingle(formData) {
        try {
            ProphitBet.showLoading();
            
            const response = await fetch('/api/predictions/single', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: formData.get('model_name'),
                    home_team: formData.get('home_team'),
                    away_team: formData.get('away_team')
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayPredictionResult(result);
            } else {
                ProphitBet.showAlert(result.error, 'danger');
                if (result.suggested_teams) {
                    this.showTeamSuggestions(result.suggested_teams);
                }
            }
            
        } catch (error) {
            ProphitBet.showAlert(`Error making prediction: ${error.message}`, 'danger');
        } finally {
            ProphitBet.hideLoading();
        }
    },
    
    // Display prediction result
    displayPredictionResult(result) {
        const resultContainer = document.getElementById('prediction-results');
        if (!resultContainer) return;
        
        const outcomeClass = result.prediction.toLowerCase().replace(' ', '-');
        
        const resultHtml = `
            <div class="prediction-result">
                <div class="prediction-header">
                    <div class="match-teams">${result.home_team} vs ${result.away_team}</div>
                    <div class="prediction-outcome outcome-${outcomeClass}">${result.prediction}</div>
                </div>
                <div class="probability-bars">
                    <div class="probability-item">
                        <div class="probability-label">Home Win</div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${result.probabilities.home_win * 100}%; background-color: var(--success-color);"></div>
                        </div>
                        <div class="probability-value">${(result.probabilities.home_win * 100).toFixed(1)}%</div>
                    </div>
                    <div class="probability-item">
                        <div class="probability-label">Draw</div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${result.probabilities.draw * 100}%; background-color: var(--warning-color);"></div>
                        </div>
                        <div class="probability-value">${(result.probabilities.draw * 100).toFixed(1)}%</div>
                    </div>
                    <div class="probability-item">
                        <div class="probability-label">Away Win</div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${result.probabilities.away_win * 100}%; background-color: var(--info-color);"></div>
                        </div>
                        <div class="probability-value">${(result.probabilities.away_win * 100).toFixed(1)}%</div>
                    </div>
                </div>
                <div class="mt-2 text-muted">
                    <small>Model: ${result.model_name} | Confidence: ${(result.confidence * 100).toFixed(1)}% | Model Accuracy: ${(result.model_accuracy * 100).toFixed(1)}%</small>
                </div>
            </div>
        `;
        
        resultContainer.innerHTML = resultHtml;
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    },
    
    // Show team suggestions
    showTeamSuggestions(teams) {
        const suggestionHtml = teams.slice(0, 10).map(team => 
            `<span class="btn btn-sm btn-secondary me-1 mb-1" onclick="PredictionManager.selectTeam('${team}')">${team}</span>`
        ).join('');
        
        ProphitBet.showAlert(`Team not found. Suggestions: <br>${suggestionHtml}`, 'warning', 10000);
    },
    
    // Select suggested team
    selectTeam(teamName) {
        const activeInput = document.querySelector('input:focus');
        if (activeInput) {
            activeInput.value = teamName;
        }
    },
    
    // Predict from CSV file
    async predictCSV(formData) {
        try {
            ProphitBet.showLoading();
            
            const response = await fetch('/api/predictions/csv', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                // Download the results file
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'predictions_results.csv';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                ProphitBet.showAlert('Predictions completed and downloaded!', 'success');
            } else {
                const result = await response.json();
                ProphitBet.showAlert(result.error, 'danger');
            }
            
        } catch (error) {
            ProphitBet.showAlert(`Error predicting CSV: ${error.message}`, 'danger');
        } finally {
            ProphitBet.hideLoading();
        }
    },
    
    // Predict fixtures
    async predictFixtures(formData) {
        try {
            ProphitBet.showLoading();
            
            const response = await fetch('/api/predictions/fixtures', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_name: formData.get('model_name'),
                    league_name: formData.get('league_name')
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayFixtureResults(result.fixtures);
            } else {
                ProphitBet.showAlert(result.error, 'danger');
            }
            
        } catch (error) {
            ProphitBet.showAlert(`Error predicting fixtures: ${error.message}`, 'danger');
        } finally {
            ProphitBet.hideLoading();
        }
    },
    
    // Display fixture results
    displayFixtureResults(fixtures) {
        const resultContainer = document.getElementById('fixture-results');
        if (!resultContainer) return;
        
        const resultsHtml = fixtures.map(fixture => {
            if (fixture.error) {
                return `
                    <div class="prediction-result">
                        <div class="prediction-header">
                            <div class="match-teams">${fixture.home_team} vs ${fixture.away_team}</div>
                            <div class="text-danger">Error</div>
                        </div>
                        <div class="text-muted">${fixture.error}</div>
                    </div>
                `;
            }
            
            const outcomeClass = fixture.prediction.toLowerCase().replace(' ', '-');
            
            return `
                <div class="prediction-result">
                    <div class="prediction-header">
                        <div class="match-teams">${fixture.home_team} vs ${fixture.away_team}</div>
                        <div class="prediction-outcome outcome-${outcomeClass}">${fixture.prediction}</div>
                    </div>
                    ${fixture.date ? `<div class="text-muted mb-2"><small>Date: ${fixture.date}</small></div>` : ''}
                    <div class="probability-bars">
                        <div class="probability-item">
                            <div class="probability-label">Home</div>
                            <div class="probability-bar">
                                <div class="probability-fill" style="width: ${fixture.probabilities.home_win * 100}%; background-color: var(--success-color);"></div>
                            </div>
                            <div class="probability-value">${(fixture.probabilities.home_win * 100).toFixed(1)}%</div>
                        </div>
                        <div class="probability-item">
                            <div class="probability-label">Draw</div>
                            <div class="probability-bar">
                                <div class="probability-fill" style="width: ${fixture.probabilities.draw * 100}%; background-color: var(--warning-color);"></div>
                            </div>
                            <div class="probability-value">${(fixture.probabilities.draw * 100).toFixed(1)}%</div>
                        </div>
                        <div class="probability-item">
                            <div class="probability-label">Away</div>
                            <div class="probability-bar">
                                <div class="probability-fill" style="width: ${fixture.probabilities.away_win * 100}%; background-color: var(--info-color);"></div>
                            </div>
                            <div class="probability-value">${(fixture.probabilities.away_win * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                    <div class="mt-2 text-muted">
                        <small>Confidence: ${(fixture.confidence * 100).toFixed(1)}%</small>
                    </div>
                </div>
            `;
        }).join('');
        
        resultContainer.innerHTML = resultsHtml;
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }
};

// Analysis Functions
const AnalysisManager = {
    // Analyze league data
    async analyzeLeague(leagueName) {
        try {
            ProphitBet.showLoading();
            
            const response = await fetch('/api/analysis/league', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ league_name: leagueName })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayLeagueAnalysis(result);
            } else {
                ProphitBet.showAlert(result.error, 'danger');
            }
            
        } catch (error) {
            ProphitBet.showAlert(`Error analyzing league: ${error.message}`, 'danger');
        } finally {
            ProphitBet.hideLoading();
        }
    },
    
    // Display league analysis
    displayLeagueAnalysis(data) {
        const container = document.getElementById('analysis-results');
        if (!container) return;
        
        let chartsHtml = '';
        
        // Add charts if available
        if (data.outcome_chart) {
            chartsHtml += `
                <div class="card mb-3">
                    <div class="card-header">
                        <h3>Match Outcome Distribution</h3>
                    </div>
                    <div class="card-body text-center">
                        <img src="data:image/png;base64,${data.outcome_chart}" alt="Outcome Distribution" class="img-fluid">
                    </div>
                </div>
            `;
        }
        
        if (data.goals_chart) {
            chartsHtml += `
                <div class="card mb-3">
                    <div class="card-header">
                        <h3>Goals per Match Distribution</h3>
                    </div>
                    <div class="card-body text-center">
                        <img src="data:image/png;base64,${data.goals_chart}" alt="Goals Distribution" class="img-fluid">
                    </div>
                </div>
            `;
        }
        
        if (data.season_chart) {
            chartsHtml += `
                <div class="card mb-3">
                    <div class="card-header">
                        <h3>Average Goals by Season</h3>
                    </div>
                    <div class="card-body text-center">
                        <img src="data:image/png;base64,${data.season_chart}" alt="Season Comparison" class="img-fluid">
                    </div>
                </div>
            `;
        }
        
        const analysisHtml = `
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h3>Basic Statistics</h3>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-sm-6">
                                    <div class="stat-card">
                                        <div class="stat-number">${data.basic_stats.total_matches}</div>
                                        <div class="stat-label">Total Matches</div>
                                    </div>
                                </div>
                                <div class="col-sm-6">
                                    <div class="stat-card">
                                        <div class="stat-number">${data.basic_stats.avg_goals_per_match}</div>
                                        <div class="stat-label">Avg Goals/Match</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h3>Outcome Distribution</h3>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-sm-4">
                                    <div class="stat-card">
                                        <div class="stat-number">${data.outcome_distribution.home_wins}</div>
                                        <div class="stat-label">Home Wins</div>
                                    </div>
                                </div>
                                <div class="col-sm-4">
                                    <div class="stat-card">
                                        <div class="stat-number">${data.outcome_distribution.draws}</div>
                                        <div class="stat-label">Draws</div>
                                    </div>
                                </div>
                                <div class="col-sm-4">
                                    <div class="stat-card">
                                        <div class="stat-number">${data.outcome_distribution.away_wins}</div>
                                        <div class="stat-label">Away Wins</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            ${chartsHtml}
        `;
        
        container.innerHTML = analysisHtml;
    },
    
    // Analyze model performance
    async analyzeModel(modelName) {
        try {
            ProphitBet.showLoading();
            
            const response = await fetch('/api/analysis/model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model_name: modelName })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayModelAnalysis(result);
            } else {
                ProphitBet.showAlert(result.error, 'danger');
            }
            
        } catch (error) {
            ProphitBet.showAlert(`Error analyzing model: ${error.message}`, 'danger');
        } finally {
            ProphitBet.hideLoading();
        }
    },
    
    // Display model analysis
    displayModelAnalysis(data) {
        const container = document.getElementById('model-analysis-results');
        if (!container) return;
        
        const performanceMetrics = data.performance_metrics || {};
        const macro_avg = performanceMetrics.macro_avg || {};
        const weighted_avg = performanceMetrics.weighted_avg || {};
        
        const analysisHtml = `
            <div class="card">
                <div class="card-header">
                    <h3>Model Performance Analysis</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-4">
                            <h4>Model Information</h4>
                            <table class="table">
                                <tr><td><strong>Name:</strong></td><td>${data.model_info.name}</td></tr>
                                <tr><td><strong>Type:</strong></td><td>${data.model_info.type}</td></tr>
                                <tr><td><strong>League:</strong></td><td>${data.model_info.league}</td></tr>
                                <tr><td><strong>Accuracy:</strong></td><td>${(data.model_info.accuracy * 100).toFixed(2)}%</td></tr>
                            </table>
                        </div>
                        <div class="col-md-4">
                            <h4>Training Information</h4>
                            <table class="table">
                                <tr><td><strong>Training Samples:</strong></td><td>${data.training_info.training_samples}</td></tr>
                                <tr><td><strong>Test Samples:</strong></td><td>${data.training_info.test_samples}</td></tr>
                                <tr><td><strong>Features:</strong></td><td>${data.training_info.features}</td></tr>
                            </table>
                        </div>
                        <div class="col-md-4">
                            <h4>Performance Metrics</h4>
                            <table class="table">
                                <tr><td><strong>Precision:</strong></td><td>${(macro_avg.precision * 100).toFixed(2)}%</td></tr>
                                <tr><td><strong>Recall:</strong></td><td>${(macro_avg.recall * 100).toFixed(2)}%</td></tr>
                                <tr><td><strong>F1-Score:</strong></td><td>${(macro_avg['f1-score'] * 100).toFixed(2)}%</td></tr>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = analysisHtml;
    }
};

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    ProphitBet.init();
});

// Export for global access
window.ProphitBet = ProphitBet;
window.LeagueManager = LeagueManager;
window.ModelManager = ModelManager;
window.PredictionManager = PredictionManager;
window.AnalysisManager = AnalysisManager;
