// Dashboard JavaScript
let currentPage = 1;
const perPage = 15;
let currentSearch = '';
let currentSegment = 'All';
let currentDonorType = 'All';

// Chart instances
let chartSegmentDonors = null;
let chartSegmentMonetary = null;
let chartCountries = null;
let chartDonorTypes = null;
let chartCampaignCategories = null;
let chartMonthlyTrends = null;

document.addEventListener('DOMContentLoaded', () => {
    initDashboard();
    initPredictorForm();
});

function initDashboard() {
    fetchSummaryKPIs();
    render3DRFMScatter();
    fetchDemographics();
    fetchCampaigns();
    loadDonorDirectory();
}

function refreshDashboard() {
    initDashboard();
}

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    event.currentTarget.classList.add('active');
    document.getElementById(`tab-${tabId}`).classList.add('active');

    if (tabId === 'overview') {
        Plotly.Plots.resize('plotly-3d-scatter');
    }
}

// Fetch Summary KPIs & Persona Cards
function fetchSummaryKPIs() {
    fetch('/api/summary')
        .then(res => res.json())
        .then(data => {
            document.getElementById('kpi-total-donors').innerText = data.total_donors.toLocaleString();
            document.getElementById('kpi-total-monetary').innerText = '$' + data.total_monetary.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            document.getElementById('kpi-avg-donation').innerText = '$' + data.avg_donation.toFixed(2);
            document.getElementById('kpi-total-txs').innerText = data.total_txs.toLocaleString();
            document.getElementById('kpi-top-country').innerText = data.top_country;

            renderSegmentCharts(data.segments);
            renderPersonaCards(data.segments);
        })
        .catch(err => console.error("Error fetching summary:", err));
}

// Render 3D RFM Scatter Plot (Plotly)
function render3DRFMScatter() {
    fetch('/api/rfm_scatter')
        .then(res => res.json())
        .then(data => {
            const traces = {};
            data.forEach(pt => {
                const seg = pt.segment_name;
                if (!traces[seg]) {
                    traces[seg] = {
                        x: [], y: [], z: [],
                        text: [],
                        mode: 'markers',
                        marker: {
                            size: 5,
                            color: pt.segment_color,
                            opacity: 0.8
                        },
                        name: seg,
                        type: 'scatter3d'
                    };
                }
                traces[seg].x.push(pt.recency);
                traces[seg].y.push(pt.frequency);
                traces[seg].z.push(pt.monetary);
                traces[seg].text.push(`Donor ID: ${pt.donor_id}<br>Country: ${pt.country}<br>Recency: ${pt.recency} days<br>Frequency: ${pt.frequency}<br>Monetary: $${pt.monetary}`);
            });

            const plotData = Object.values(traces);

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#9CA3AF', family: 'Inter' },
                margin: { l: 0, r: 0, b: 0, t: 30 },
                scene: {
                    xaxis: { title: 'Recency (Days)', backgroundcolor: 'rgba(0,0,0,0)', gridcolor: 'rgba(255,255,255,0.1)' },
                    yaxis: { title: 'Frequency (Count)', backgroundcolor: 'rgba(0,0,0,0)', gridcolor: 'rgba(255,255,255,0.1)' },
                    zaxis: { title: 'Monetary ($)', backgroundcolor: 'rgba(0,0,0,0)', gridcolor: 'rgba(255,255,255,0.1)' },
                    camera: { eye: { x: 1.4, y: 1.4, z: 1.2 } }
                },
                legend: { orientation: 'h', x: 0, y: 1 }
            };

            Plotly.newPlot('plotly-3d-scatter', plotData, layout, {responsive: true});
        })
        .catch(err => console.error("Error fetching 3D scatter data:", err));
}

// Render Segment Charts
function renderSegmentCharts(segments) {
    const labels = segments.map(s => s.name);
    const colors = segments.map(s => s.color);
    const counts = segments.map(s => s.count);
    const monetary = segments.map(s => s.total_monetary);

    // Segment Donors Pie
    if (chartSegmentDonors) chartSegmentDonors.destroy();
    const ctx1 = document.getElementById('chart-segment-donors').getContext('2d');
    chartSegmentDonors = new Chart(ctx1, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#0B0F19'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#9CA3AF' } }
            }
        }
    });

    // Segment Monetary Bar
    if (chartSegmentMonetary) chartSegmentMonetary.destroy();
    const ctx2 = document.getElementById('chart-segment-monetary').getContext('2d');
    chartSegmentMonetary = new Chart(ctx2, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Monetary ($)',
                data: monetary,
                backgroundColor: colors,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: '#9CA3AF' }, grid: { display: false } },
                y: { ticks: { color: '#9CA3AF' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// Render Persona Cards
function renderPersonaCards(segments) {
    const container = document.getElementById('persona-cards-container');
    container.innerHTML = '';

    segments.forEach(seg => {
        const card = document.createElement('div');
        card.className = 'persona-card';
        card.style.borderLeftColor = seg.color;

        card.innerHTML = `
            <div class="persona-card-header">
                <span class="persona-name" style="color: ${seg.color}">${seg.name}</span>
                <span class="badge" style="background: ${seg.color}22; color: ${seg.color}; border: 1px solid ${seg.color}">${seg.count} Donors (${seg.pct_donors}%)</span>
            </div>
            <div class="persona-metrics">
                <div>
                    <div class="metric-val">${Math.round(seg.avg_recency)}d</div>
                    <div class="metric-lbl">Avg Recency</div>
                </div>
                <div>
                    <div class="metric-val">${seg.avg_frequency.toFixed(1)}</div>
                    <div class="metric-lbl">Avg Frequency</div>
                </div>
                <div>
                    <div class="metric-val">$${seg.avg_monetary.toFixed(0)}</div>
                    <div class="metric-lbl">Avg Monetary</div>
                </div>
            </div>
            <div class="persona-strategy">
                <strong><i class="fa-solid fa-bullseye"></i> Engagement Strategy:</strong> ${seg.strategy}
            </div>
        `;
        container.appendChild(card);
    });
}

// Fetch Demographics
function fetchDemographics() {
    fetch('/api/demographics')
        .then(res => res.json())
        .then(data => {
            // Country Chart
            const countryLabels = Object.keys(data.countries);
            const countryVals = Object.values(data.countries);

            if (chartCountries) chartCountries.destroy();
            const ctx1 = document.getElementById('chart-countries').getContext('2d');
            chartCountries = new Chart(ctx1, {
                type: 'bar',
                data: {
                    labels: countryLabels,
                    datasets: [{
                        label: 'Donors Count',
                        data: countryVals,
                        backgroundColor: '#3B82F6',
                        borderRadius: 6
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: '#9CA3AF' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { ticks: { color: '#9CA3AF' }, grid: { display: false } }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            // Donor Type Chart
            const typeLabels = Object.keys(data.donor_types);
            const typeVals = Object.values(data.donor_types);

            if (chartDonorTypes) chartDonorTypes.destroy();
            const ctx2 = document.getElementById('chart-donor-types').getContext('2d');
            chartDonorTypes = new Chart(ctx2, {
                type: 'pie',
                data: {
                    labels: typeLabels,
                    datasets: [{
                        data: typeVals,
                        backgroundColor: ['#10B981', '#3B82F6', '#F59E0B']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom', labels: { color: '#9CA3AF' } } }
                }
            });
        });
}

// Fetch Campaigns & Trends
function fetchCampaigns() {
    fetch('/api/campaigns')
        .then(res => res.json())
        .then(data => {
            // Campaign Categories
            const catLabels = data.categories.map(c => c.category);
            const catAmounts = data.categories.map(c => c.total_raised);

            if (chartCampaignCategories) chartCampaignCategories.destroy();
            const ctx1 = document.getElementById('chart-campaign-categories').getContext('2d');
            chartCampaignCategories = new Chart(ctx1, {
                type: 'bar',
                data: {
                    labels: catLabels,
                    datasets: [{
                        label: 'Funds Raised ($)',
                        data: catAmounts,
                        backgroundColor: '#10B981',
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: '#9CA3AF' }, grid: { display: false } },
                        y: { ticks: { color: '#9CA3AF' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            // Monthly Trends
            const trendMonths = data.monthly_trends.map(t => t.month);
            const trendAmounts = data.monthly_trends.map(t => t.total_raised);

            if (chartMonthlyTrends) chartMonthlyTrends.destroy();
            const ctx2 = document.getElementById('chart-monthly-trends').getContext('2d');
            chartMonthlyTrends = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: trendMonths,
                    datasets: [{
                        label: 'Monthly Raised ($)',
                        data: trendAmounts,
                        borderColor: '#F59E0B',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: '#9CA3AF', maxRotation: 45 }, grid: { display: false } },
                        y: { ticks: { color: '#9CA3AF' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        });
}

// Load Directory Table
function loadDonorDirectory() {
    currentSearch = document.getElementById('search-input').value.trim();
    currentSegment = document.getElementById('filter-segment').value;
    currentDonorType = document.getElementById('filter-donor-type').value;

    const url = `/api/donors?page=${currentPage}&per_page=${perPage}&search=${encodeURIComponent(currentSearch)}&segment=${encodeURIComponent(currentSegment)}&donor_type=${encodeURIComponent(currentDonorType)}`;

    fetch(url)
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('donor-table-body');
            tbody.innerHTML = '';

            data.records.forEach(donor => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>#${donor.donor_id}</strong></td>
                    <td>${donor.age} yrs (${donor.gender})</td>
                    <td>${donor.country}</td>
                    <td><span class="badge info">${donor.donor_type}</span></td>
                    <td>${donor.recency} days</td>
                    <td>${donor.frequency} txs</td>
                    <td><strong>$${donor.monetary.toFixed(2)}</strong></td>
                    <td>$${donor.avg_donation.toFixed(2)}</td>
                    <td><code>${donor.rfm_score}</code></td>
                    <td><span class="badge" style="background: ${donor.segment_color}22; color: ${donor.segment_color}; border: 1px solid ${donor.segment_color}">${donor.segment_name}</span></td>
                    <td>
                        <button class="btn btn-sm btn-outline" onclick="openDonorModal(${donor.donor_id})">
                            <i class="fa-solid fa-eye"></i> View
                        </button>
                    </td>
                `;
                tbody.appendChild(tr);
            });

            document.getElementById('pagination-info').innerText = `Showing ${(currentPage-1)*perPage + 1} to ${Math.min(currentPage*perPage, data.total_count)} of ${data.total_count} donors`;
            document.getElementById('current-page-num').innerText = currentPage;

            document.getElementById('btn-prev-page').disabled = (currentPage <= 1);
            document.getElementById('btn-next-page').disabled = (currentPage >= data.total_pages);
        });
}

function filterDirectory() {
    currentPage = 1;
    loadDonorDirectory();
}

function changePage(delta) {
    currentPage += delta;
    if (currentPage < 1) currentPage = 1;
    loadDonorDirectory();
}

// Open Donor Modal
function openDonorModal(donorId) {
    fetch(`/api/donor/${donorId}`)
        .then(res => res.json())
        .then(donor => {
            const body = document.getElementById('modal-body');
            let txRows = donor.transactions.map(t => `
                <tr>
                    <td>#${t.transaction_id}</td>
                    <td>${t.date}</td>
                    <td>$${t.amount.toFixed(2)}</td>
                    <td>${t.payment_method}</td>
                    <td>${t.aid_country} (${t.aid_sector})</td>
                </tr>
            `).join('');

            body.innerHTML = `
                <div class="modal-profile-header">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                        <h2>Donor #${donor.donor_id}</h2>
                        <span class="badge" style="background: ${donor.segment_color}22; color: ${donor.segment_color}; font-size: 14px; padding: 6px 14px; border: 1px solid ${donor.segment_color}">${donor.segment_name} (${donor.segment_badge})</span>
                    </div>
                </div>

                <div class="persona-metrics" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 20px;">
                    <div><div class="metric-val">${donor.recency} days</div><div class="metric-lbl">Recency</div></div>
                    <div><div class="metric-val">${donor.frequency}</div><div class="metric-lbl">Frequency</div></div>
                    <div><div class="metric-val">$${donor.monetary.toFixed(2)}</div><div class="metric-lbl">Monetary</div></div>
                    <div><div class="metric-val">${donor.rfm_score}</div><div class="metric-lbl">RFM Code</div></div>
                </div>

                <div class="glass-card" style="padding: 16px; margin-bottom: 20px; border-left: 4px solid ${donor.segment_color}">
                    <h4 style="color: ${donor.segment_color}; margin-bottom: 6px;"><i class="fa-solid fa-bullseye"></i> Engagement Strategy:</h4>
                    <p style="color: #9CA3AF; font-size: 14px;">${donor.segment_strategy}</p>
                </div>

                <h4 style="color: #FFF; margin-bottom: 12px;"><i class="fa-solid fa-history"></i> Recent Transaction History</h4>
                <div class="table-responsive">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Tx ID</th>
                                <th>Date</th>
                                <th>Amount</th>
                                <th>Payment Method</th>
                                <th>Aid Location & Sector</th>
                            </tr>
                        </thead>
                        <tbody>${txRows}</tbody>
                    </table>
                </div>
            `;
            document.getElementById('donor-modal').classList.remove('hidden');
        });
}

function closeModal() {
    document.getElementById('donor-modal').classList.add('hidden');
}

// Export CSV
function exportDirectoryCSV() {
    window.location.href = '/api/donors?page=1&per_page=5000';
}

// Predictor Logic
function togglePredictMode() {
    const mode = document.querySelector('input[name="predict_mode"]:checked').value;
    if (mode === 'lookup') {
        document.getElementById('lookup-group').classList.remove('hidden');
        document.getElementById('custom-group').classList.add('hidden');
    } else {
        document.getElementById('lookup-group').classList.add('hidden');
        document.getElementById('custom-group').classList.remove('hidden');
    }
}

function handlePredictSubmit(e) {
    e.preventDefault();
    const mode = document.querySelector('input[name="predict_mode"]:checked').value;
    let payload = {};

    if (mode === 'lookup') {
        payload.donor_id = document.getElementById('input-donor-id').value;
    } else {
        payload.recency = document.getElementById('input-recency').value;
        payload.frequency = document.getElementById('input-frequency').value;
        payload.monetary = document.getElementById('input-monetary').value;
    }

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(res => {
        const box = document.getElementById('prediction-output-box');
        box.innerHTML = `
            <div class="prediction-result-box">
                <div class="pred-header" style="background: ${res.segment_color}18; border: 1px solid ${res.segment_color}">
                    <div style="font-size: 32px; color: ${res.segment_color}"><i class="fa-solid fa-brain"></i></div>
                    <div>
                        <div class="pred-badge" style="color: ${res.segment_color}">${res.predicted_segment}</div>
                        <span class="badge" style="background: ${res.segment_color}33; color: ${res.segment_color}">${res.segment_badge}</span>
                    </div>
                </div>

                <div style="background: rgba(0,0,0,0.2); padding: 16px; border-radius: 10px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px; color: #9CA3AF;">
                        <span>K-Means Model Confidence:</span>
                        <span style="color: #FFF; font-weight: 700;">${res.confidence_score}% Match</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${res.confidence_score}%; background: ${res.segment_color}"></div>
                    </div>
                </div>

                <div class="persona-metrics" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 16px;">
                    <div><div class="metric-val">${res.recency_input} days</div><div class="metric-lbl">Recency</div></div>
                    <div><div class="metric-val">${res.frequency_input}</div><div class="metric-lbl">Frequency</div></div>
                    <div><div class="metric-val">$${res.monetary_input}</div><div class="metric-lbl">Monetary Total</div></div>
                </div>

                <div class="glass-card" style="padding: 16px; border-left: 4px solid ${res.segment_color}">
                    <h4 style="color: ${res.segment_color}; margin-bottom: 6px;"><i class="fa-solid fa-lightbulb"></i> Strategic Action Recommendation:</h4>
                    <p style="color: #9CA3AF; font-size: 14px;">${res.segment_strategy}</p>
                </div>
            </div>
        `;
    });
}
