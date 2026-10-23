// Mobile sidebar toggle
document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.querySelector('.mobile-nav-toggle');
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.createElement('div');
    overlay.className = 'sidebar-overlay';
    overlay.style.cssText = 'display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:999;';
    document.body.appendChild(overlay);

    if (toggle) {
        toggle.addEventListener('click', function() {
            sidebar.classList.toggle('show');
            overlay.style.display = sidebar.classList.contains('show') ? 'block' : 'none';
        });
    }

    overlay.addEventListener('click', function() {
        sidebar.classList.remove('show');
        overlay.style.display = 'none';
    });
});

// Utility function for fetch with error handling
async function fetchJSON(url, options = {}) {
    try {
        const response = await fetch(url, options);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `HTTP error ${response.status}`);
        }
        return data;
    } catch (error) {
        console.error('Fetch error:', error);
        throw error;
    }
}

// Show/hide loading spinner
function showLoading(containerId) {
    const el = document.getElementById(containerId);
    if (el) el.classList.add('active');
}

function hideLoading(containerId) {
    const el = document.getElementById(containerId);
    if (el) el.classList.remove('active');
}

// Show alert message
function showAlert(containerId, message, type = 'danger') {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>`;
    }
}

// Chart color palette
const chartColors = {
    primary: '#2c5282',
    secondary: '#3182ce',
    success: '#38a169',
    danger: '#e53e3e',
    warning: '#dd6b20',
    info: '#00b5d8',
    purple: '#805ad5',
    pink: '#d53f8c',
    palette: ['#2c5282', '#38a169', '#e53e3e', '#dd6b20', '#805ad5', '#00b5d8', '#d53f8c', '#319795']
};

// Default chart options
const defaultChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: {
                font: { family: 'Segoe UI, system-ui, sans-serif', size: 12 },
                padding: 16
            }
        }
    },
    scales: {
        x: {
            ticks: { font: { size: 11 } },
            grid: { color: 'rgba(0,0,0,0.05)' }
        },
        y: {
            ticks: { font: { size: 11 } },
            grid: { color: 'rgba(0,0,0,0.05)' }
        }
    }
};
