document.addEventListener('DOMContentLoaded', function() {
    loadAnalyticsData();
});

let kmRiskChartInstance = null;
let kmStageChartInstance = null;

async function loadAnalyticsData() {
    try {
        showLoading('analyticsLoading');
        const data = await fetchJSON('/api/analytics');
        hideLoading('analyticsLoading');
        renderAnalytics(data);
    } catch (error) {
        hideLoading('analyticsLoading');
        showAlert('analyticsAlerts', 'Failed to load analytics data: ' + error.message);
    }
}

function renderAnalytics(data) {
    // KM by Risk Groups
    if (data.km_data && data.km_data.groups) {
        const ctx = document.getElementById('kmRiskChart');
        if (ctx) {
            if (kmRiskChartInstance) kmRiskChartInstance.destroy();
            const datasets = [];
            const colors = { 'High Risk': '#e53e3e', 'Low Risk': '#38a169' };
            
            for (const [groupName, groupData] of Object.entries(data.km_data.groups)) {
                datasets.push({
                    label: groupName + ' (n=' + groupData.n + ')',
                    data: groupData.timeline.map((t, i) => ({ x: t, y: groupData.survival_prob[i] })),
                    borderColor: colors[groupName] || chartColors.palette[datasets.length],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    stepped: 'before',
                    pointRadius: 0,
                    tension: 0
                });
            }
            
            kmRiskChartInstance = new Chart(ctx, {
                type: 'line',
                data: { datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Time (months)' },
                            min: 0,
                            grid: { color: 'rgba(0,0,0,0.05)' }
                        },
                        y: {
                            title: { display: true, text: 'Survival Probability' },
                            min: 0, max: 1,
                            grid: { color: 'rgba(0,0,0,0.05)' }
                        }
                    },
                    plugins: {
                        title: { display: true, text: 'Kaplan-Meier Survival Curves by Risk Group', font: { size: 14, weight: 'bold' } },
                        legend: { position: 'bottom' }
                    }
                }
            });
        }
        
        // Log-rank test
        if (data.km_data.logrank_p !== undefined) {
            const lrEl = document.getElementById('logrankResult');
            if (lrEl) {
                const pVal = data.km_data.logrank_p;
                const sig = pVal < 0.05 ? 'statistically significant' : 'not statistically significant';
                lrEl.innerHTML = `
                    <div class="alert alert-info">
                        <strong>Log-Rank Test:</strong> p-value = ${pVal.toFixed(4)} (${sig} at α=0.05)<br>
                        <small class="text-muted">Note: Statistical significance in synthetic data does not imply clinical relevance.</small>
                    </div>`;
            }
        }
    } else if (data.km_error) {
        showAlert('analyticsAlerts', 'KM Analysis: ' + data.km_error, 'warning');
    }
    
    // KM by Tumor Stage
    if (data.km_by_stage && data.km_by_stage.groups) {
        const ctx = document.getElementById('kmStageChart');
        if (ctx) {
            if (kmStageChartInstance) kmStageChartInstance.destroy();
            const datasets = [];
            const stageColors = { 'I': '#38a169', 'II': '#2c5282', 'III': '#dd6b20', 'IV': '#e53e3e' };
            
            for (const [stage, stageData] of Object.entries(data.km_by_stage.groups)) {
                datasets.push({
                    label: 'Stage ' + stage + ' (n=' + stageData.n + ')',
                    data: stageData.timeline.map((t, i) => ({ x: t, y: stageData.survival_prob[i] })),
                    borderColor: stageColors[stage] || chartColors.palette[datasets.length],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    stepped: 'before',
                    pointRadius: 0,
                    tension: 0
                });
            }
            
            kmStageChartInstance = new Chart(ctx, {
                type: 'line',
                data: { datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Time (months)' },
                            min: 0,
                            grid: { color: 'rgba(0,0,0,0.05)' }
                        },
                        y: {
                            title: { display: true, text: 'Survival Probability' },
                            min: 0, max: 1,
                            grid: { color: 'rgba(0,0,0,0.05)' }
                        }
                    },
                    plugins: {
                        title: { display: true, text: 'Kaplan-Meier Survival Curves by Tumor Stage', font: { size: 14, weight: 'bold' } },
                        legend: { position: 'bottom' }
                    }
                }
            });
        }
        
        if (data.km_by_stage.logrank_p !== undefined) {
            const lrEl = document.getElementById('logrankStageResult');
            if (lrEl) {
                const pVal = data.km_by_stage.logrank_p;
                lrEl.innerHTML = `
                    <div class="alert alert-info">
                        <strong>Log-Rank Test (Stages):</strong> p-value = ${pVal.toFixed(4)}<br>
                        <small class="text-muted">Comparing survival curves across tumor stages in synthetic data.</small>
                    </div>`;
            }
        }
    } else if (data.km_stage_error) {
        showAlert('analyticsAlerts', 'Stage KM: ' + data.km_stage_error, 'warning');
    }
}
