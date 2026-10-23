document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();
});

async function loadDashboardData() {
    try {
        showLoading('dashboardLoading');
        const data = await fetchJSON('/api/dashboard');
        hideLoading('dashboardLoading');
        renderDashboardCharts(data);
    } catch (error) {
        hideLoading('dashboardLoading');
        showAlert('dashboardAlerts', 'Failed to load dashboard data: ' + error.message);
    }
}

function renderDashboardCharts(data) {
    // Update stat cards
    if (data.basic_stats) {
        const els = {
            'statTotal': data.basic_stats.total_patients,
            'statAvgAge': data.basic_stats.avg_age,
            'statAvgSurv': data.basic_stats.avg_survival,
            'statMedianSurv': data.basic_stats.median_survival
        };
        for (const [id, val] of Object.entries(els)) {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        }
    }

    // Age Distribution Chart
    if (data.age_distribution) {
        const ctx = document.getElementById('ageDistChart');
        if (ctx) {
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.age_distribution.labels,
                    datasets: [{
                        label: 'Number of Patients',
                        data: data.age_distribution.values,
                        backgroundColor: 'rgba(44, 82, 130, 0.7)',
                        borderColor: '#2c5282',
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    ...defaultChartOptions,
                    plugins: { ...defaultChartOptions.plugins, title: { display: true, text: 'Age Distribution', font: { size: 14, weight: 'bold' } } }
                }
            });
        }
    }

    // Tumor Stage Distribution
    if (data.tumor_stage_distribution) {
        const ctx = document.getElementById('tumorStageChart');
        if (ctx) {
            const stages = Object.keys(data.tumor_stage_distribution).sort();
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: stages.map(s => 'Stage ' + s),
                    datasets: [{
                        data: stages.map(s => data.tumor_stage_distribution[s]),
                        backgroundColor: chartColors.palette.slice(0, stages.length),
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { display: true, text: 'Tumor Stage Distribution', font: { size: 14, weight: 'bold' } },
                        legend: { position: 'bottom' }
                    }
                }
            });
        }
    }

    // Event Distribution
    if (data.event_distribution) {
        const ctx = document.getElementById('eventDistChart');
        if (ctx) {
            new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Events (Observed)', 'Censored'],
                    datasets: [{
                        data: [data.event_distribution.events, data.event_distribution.censored],
                        backgroundColor: ['#e53e3e', '#38a169'],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: { display: true, text: 'Event Distribution', font: { size: 14, weight: 'bold' } },
                        legend: { position: 'bottom' }
                    }
                }
            });
        }
    }

    // Survival Distribution
    if (data.survival_distribution) {
        const ctx = document.getElementById('survivalDistChart');
        if (ctx) {
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.survival_distribution.labels,
                    datasets: [{
                        label: 'Number of Patients',
                        data: data.survival_distribution.values,
                        backgroundColor: 'rgba(56, 161, 105, 0.7)',
                        borderColor: '#38a169',
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    ...defaultChartOptions,
                    plugins: { ...defaultChartOptions.plugins, title: { display: true, text: 'Survival Time Distribution (months)', font: { size: 14, weight: 'bold' } } }
                }
            });
        }
    }

    // Gene Expression Chart
    if (data.gene_expression_stats) {
        const ctx = document.getElementById('geneExprChart');
        if (ctx) {
            const genes = Object.keys(data.gene_expression_stats);
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: genes.map(g => g.replace('gene_expr_', 'Gene ')),
                    datasets: [
                        {
                            label: 'Mean',
                            data: genes.map(g => data.gene_expression_stats[g].mean),
                            backgroundColor: 'rgba(44, 82, 130, 0.7)',
                            borderRadius: 4
                        },
                        {
                            label: 'Median',
                            data: genes.map(g => data.gene_expression_stats[g].median),
                            backgroundColor: 'rgba(49, 130, 206, 0.7)',
                            borderRadius: 4
                        }
                    ]
                },
                options: {
                    ...defaultChartOptions,
                    plugins: { ...defaultChartOptions.plugins, title: { display: true, text: 'Gene Expression Summary', font: { size: 14, weight: 'bold' } } }
                }
            });
        }
    }

    // Methylation Chart
    if (data.methylation_stats) {
        const ctx = document.getElementById('methylationChart');
        if (ctx) {
            const meths = Object.keys(data.methylation_stats);
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: meths.map(m => m.replace('methylation_', 'Methylation ')),
                    datasets: [
                        {
                            label: 'Mean',
                            data: meths.map(m => data.methylation_stats[m].mean),
                            backgroundColor: 'rgba(128, 90, 213, 0.7)',
                            borderRadius: 4
                        },
                        {
                            label: 'Std Dev',
                            data: meths.map(m => data.methylation_stats[m].std),
                            backgroundColor: 'rgba(213, 63, 140, 0.7)',
                            borderRadius: 4
                        }
                    ]
                },
                options: {
                    ...defaultChartOptions,
                    plugins: { ...defaultChartOptions.plugins, title: { display: true, text: 'DNA Methylation Summary', font: { size: 14, weight: 'bold' } } }
                }
            });
        }
    }

    // Model metrics
    if (data.model_metrics) {
        const container = document.getElementById('modelMetrics');
        if (container) {
            let html = '<div class="row">';
            for (const [name, metrics] of Object.entries(data.model_metrics)) {
                html += `
                    <div class="col-md-6">
                        <div class="stat-card">
                            <div class="stat-label">${name.toUpperCase()}</div>
                            <div class="stat-value">${metrics.c_index ? metrics.c_index.toFixed(4) : 'N/A'}</div>
                            <div class="stat-label">C-Index</div>
                        </div>
                    </div>`;
            }
            html += '</div>';
            container.innerHTML = html;
        }
    }
}
