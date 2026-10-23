document.addEventListener('DOMContentLoaded', function() {
    loadGlobalExplainability();
    
    const form = document.getElementById('localExplainForm');
    if (form) {
        form.addEventListener('submit', handleLocalExplanation);
    }
    
    const sampleBtn = document.getElementById('explainSampleBtn');
    if (sampleBtn) {
        sampleBtn.addEventListener('click', fillExplainSample);
    }
});

let globalImportChart = null;
let localContribChartInstance = null;

async function loadGlobalExplainability() {
    try {
        showLoading('explainLoading');
        const data = await fetchJSON('/api/explainability');
        hideLoading('explainLoading');
        renderGlobalImportance(data);
        renderFeatureDescriptions(data.feature_descriptions);
    } catch (error) {
        hideLoading('explainLoading');
        showAlert('explainAlerts', 'Failed to load explainability data: ' + error.message);
    }
}

function renderGlobalImportance(data) {
    if (!data.global_importance) {
        if (data.global_importance_error) {
            showAlert('explainAlerts', 'Global importance: ' + data.global_importance_error, 'warning');
        }
        return;
    }
    
    const imp = data.global_importance;
    const methodLabel = document.getElementById('importanceMethodLabel');
    if (methodLabel && imp.method) {
        methodLabel.textContent = 'Method: ' + imp.method;
    }
    
    const ctx = document.getElementById('globalImportanceChart');
    if (ctx && imp.features && imp.importances) {
        if (globalImportChart) globalImportChart.destroy();
        globalImportChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: imp.features,
                datasets: [{
                    label: 'Feature Importance',
                    data: imp.importances,
                    backgroundColor: 'rgba(44, 82, 130, 0.7)',
                    borderColor: '#2c5282',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                ...defaultChartOptions,
                indexAxis: 'y',
                plugins: {
                    ...defaultChartOptions.plugins,
                    title: { display: true, text: 'Global Feature Importance', font: { size: 14, weight: 'bold' } },
                    legend: { display: false }
                }
            }
        });
    }
}

function renderFeatureDescriptions(descriptions) {
    if (!descriptions) return;
    const tbody = document.getElementById('featureDescBody');
    if (!tbody) return;
    
    let html = '';
    for (const [feature, desc] of Object.entries(descriptions)) {
        html += `<tr><td><code>${feature}</code></td><td>${desc}</td></tr>`;
    }
    tbody.innerHTML = html;
}

function fillExplainSample() {
    document.getElementById('explain_age').value = 62;
    document.getElementById('explain_tumor_stage').value = 'III';
    document.getElementById('explain_gene_expr_A').value = 9.5;
    document.getElementById('explain_gene_expr_B').value = 4.2;
    document.getElementById('explain_gene_expr_C').value = 7.1;
    document.getElementById('explain_methylation_A').value = 0.65;
    document.getElementById('explain_methylation_B').value = 0.42;
}

async function handleLocalExplanation(e) {
    e.preventDefault();
    
    const btn = document.getElementById('explainBtn');
    const spinner = document.getElementById('explainSpinner');
    btn.disabled = true;
    spinner.style.display = 'inline-block';
    document.getElementById('localExplainAlerts').innerHTML = '';
    
    const payload = {
        patient: {
            age: parseFloat(document.getElementById('explain_age').value),
            tumor_stage: document.getElementById('explain_tumor_stage').value,
            gene_expr_A: parseFloat(document.getElementById('explain_gene_expr_A').value),
            gene_expr_B: parseFloat(document.getElementById('explain_gene_expr_B').value),
            gene_expr_C: parseFloat(document.getElementById('explain_gene_expr_C').value),
            methylation_A: parseFloat(document.getElementById('explain_methylation_A').value),
            methylation_B: parseFloat(document.getElementById('explain_methylation_B').value),
        },
        model: document.getElementById('explain_model').value
    };
    
    try {
        const data = await fetchJSON('/api/explainability', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        
        if (data.local_explanation) {
            renderLocalExplanation(data.local_explanation);
        } else if (data.local_explanation_error) {
            showAlert('localExplainAlerts', data.local_explanation_error, 'warning');
        }
    } catch (error) {
        showAlert('localExplainAlerts', error.message, 'danger');
    } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
    }
}

function renderLocalExplanation(explanation) {
    const container = document.getElementById('localExplainResult');
    if (!container) return;
    container.style.display = 'block';
    
    const methodEl = document.getElementById('localMethodLabel');
    if (methodEl && explanation.method) {
        methodEl.textContent = 'Method: ' + explanation.method;
    }
    
    if (explanation.features && explanation.contributions) {
        const ctx = document.getElementById('localContribChart');
        if (ctx) {
            if (localContribChartInstance) localContribChartInstance.destroy();
            
            // Sort by absolute value
            const pairs = explanation.features.map((f, i) => [f, explanation.contributions[i]]);
            pairs.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
            
            localContribChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: pairs.map(p => p[0]),
                    datasets: [{
                        label: 'Contribution',
                        data: pairs.map(p => p[1]),
                        backgroundColor: pairs.map(p => p[1] >= 0 ? 'rgba(229, 62, 62, 0.7)' : 'rgba(56, 161, 105, 0.7)'),
                        borderColor: pairs.map(p => p[1] >= 0 ? '#e53e3e' : '#38a169'),
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    ...defaultChartOptions,
                    indexAxis: 'y',
                    plugins: {
                        ...defaultChartOptions.plugins,
                        title: {
                            display: true,
                            text: 'Local Feature Contributions (Red = ↑Risk, Green = ↓Risk)',
                            font: { size: 13, weight: 'bold' }
                        },
                        legend: { display: false }
                    }
                }
            });
        }
    }
}
