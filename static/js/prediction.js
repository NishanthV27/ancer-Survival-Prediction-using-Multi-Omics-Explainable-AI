document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictForm');
    const sampleBtn = document.getElementById('sampleBtn');
    const clearBtn = document.getElementById('clearBtn');

    if (form) {
        form.addEventListener('submit', handlePrediction);
    }
    if (sampleBtn) {
        sampleBtn.addEventListener('click', fillSampleData);
    }
    if (clearBtn) {
        clearBtn.addEventListener('click', clearForm);
    }
});

function fillSampleData() {
    document.getElementById('age').value = 62;
    document.getElementById('tumor_stage').value = 'III';
    document.getElementById('gene_expr_A').value = 9.5;
    document.getElementById('gene_expr_B').value = 4.2;
    document.getElementById('gene_expr_C').value = 7.1;
    document.getElementById('methylation_A').value = 0.65;
    document.getElementById('methylation_B').value = 0.42;
}

function clearForm() {
    document.getElementById('predictForm').reset();
    document.getElementById('resultSection').classList.remove('show');
    document.getElementById('predictionAlerts').innerHTML = '';
}

let survProbChart = null;
let featureContribChart = null;

async function handlePrediction(e) {
    e.preventDefault();
    
    const resultSection = document.getElementById('resultSection');
    resultSection.classList.remove('show');
    document.getElementById('predictionAlerts').innerHTML = '';
    
    const submitBtn = document.getElementById('submitBtn');
    const spinner = document.getElementById('predictSpinner');
    submitBtn.disabled = true;
    spinner.style.display = 'inline-block';
    
    const payload = {
        age: parseFloat(document.getElementById('age').value),
        tumor_stage: document.getElementById('tumor_stage').value,
        gene_expr_A: parseFloat(document.getElementById('gene_expr_A').value),
        gene_expr_B: parseFloat(document.getElementById('gene_expr_B').value),
        gene_expr_C: parseFloat(document.getElementById('gene_expr_C').value),
        methylation_A: parseFloat(document.getElementById('methylation_A').value),
        methylation_B: parseFloat(document.getElementById('methylation_B').value),
        model: document.getElementById('modelSelect').value
    };
    
    try {
        const result = await fetchJSON('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        
        displayPredictionResult(result);
    } catch (error) {
        showAlert('predictionAlerts', error.message, 'danger');
    } finally {
        submitBtn.disabled = false;
        spinner.style.display = 'none';
    }
}

function displayPredictionResult(result) {
    const resultSection = document.getElementById('resultSection');
    
    // Risk score and category
    const riskScoreEl = document.getElementById('riskScoreValue');
    const riskBadgeEl = document.getElementById('riskBadge');
    const riskExplanation = document.getElementById('riskExplanation');
    
    if (riskScoreEl) {
        riskScoreEl.textContent = result.risk_score !== undefined ? result.risk_score.toFixed(4) : 'N/A';
        riskScoreEl.className = 'risk-score-value ' + 
            (result.risk_category === 'High Risk' ? 'text-danger' : 'text-success');
    }
    
    if (riskBadgeEl) {
        riskBadgeEl.textContent = result.risk_category || 'Unknown';
        riskBadgeEl.className = 'risk-badge ' + 
            (result.risk_category === 'High Risk' ? 'high-risk' : 'low-risk');
    }
    
    if (riskExplanation) {
        riskExplanation.innerHTML = result.explanation || '';
    }
    
    // Survival probability chart
    if (result.survival_probabilities && Object.keys(result.survival_probabilities).length > 0) {
        document.getElementById('survProbContainer').style.display = 'block';
        const ctx = document.getElementById('survivalProbChart');
        if (ctx) {
            if (survProbChart) survProbChart.destroy();
            const timePoints = Object.keys(result.survival_probabilities).map(Number).sort((a,b) => a-b);
            survProbChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: timePoints.map(t => t + ' months'),
                    datasets: [{
                        label: 'Survival Probability',
                        data: timePoints.map(t => result.survival_probabilities[t]),
                        borderColor: '#2c5282',
                        backgroundColor: 'rgba(44, 82, 130, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 6,
                        pointBackgroundColor: '#2c5282',
                        borderWidth: 2
                    }]
                },
                options: {
                    ...defaultChartOptions,
                    scales: {
                        ...defaultChartOptions.scales,
                        y: { ...defaultChartOptions.scales.y, min: 0, max: 1, title: { display: true, text: 'Probability' } },
                        x: { ...defaultChartOptions.scales.x, title: { display: true, text: 'Time' } }
                    },
                    plugins: {
                        ...defaultChartOptions.plugins,
                        title: { display: true, text: 'Estimated Survival Probability', font: { size: 14, weight: 'bold' } }
                    }
                }
            });
        }
    } else {
        document.getElementById('survProbContainer').style.display = 'none';
    }
    
    // Feature contributions chart
    if (result.feature_contributions) {
        document.getElementById('featureContribContainer').style.display = 'block';
        const ctx = document.getElementById('featureContribChart');
        if (ctx) {
            if (featureContribChart) featureContribChart.destroy();
            const features = Object.entries(result.feature_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
            featureContribChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: features.map(f => f[0]),
                    datasets: [{
                        label: 'Contribution to Risk',
                        data: features.map(f => f[1]),
                        backgroundColor: features.map(f => f[1] >= 0 ? 'rgba(229, 62, 62, 0.7)' : 'rgba(56, 161, 105, 0.7)'),
                        borderColor: features.map(f => f[1] >= 0 ? '#e53e3e' : '#38a169'),
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    ...defaultChartOptions,
                    indexAxis: 'y',
                    plugins: {
                        ...defaultChartOptions.plugins,
                        title: { display: true, text: 'Feature Contributions (Red = ↑Risk, Green = ↓Risk)', font: { size: 13, weight: 'bold' } },
                        legend: { display: false }
                    }
                }
            });
        }
    } else {
        document.getElementById('featureContribContainer').style.display = 'none';
    }
    
    resultSection.classList.add('show');
}
