document.getElementById('trainBtn').addEventListener('click', async () => {
    const btn = document.getElementById('trainBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    
    btn.disabled = true;
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    
    try {
        const response = await fetch('/api/train');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        
        renderChart(data.q_learning.rewards, data.sarsa.rewards);
        renderGrid('qGrid', data.q_learning.policy);
        renderGrid('sarsaGrid', data.sarsa.policy);
        
        results.classList.remove('hidden');
    } catch (error) {
        console.error("Error training:", error);
        alert("Failed to train agents. Make sure the backend is running and '/api/train' is accessible.");
    } finally {
        btn.disabled = false;
        loading.classList.add('hidden');
    }
});

let rewardChart = null;

function renderChart(qRewards, sarsaRewards) {
    const ctx = document.getElementById('rewardChart').getContext('2d');
    
    if (rewardChart) {
        rewardChart.destroy();
    }
    
    const labels = Array.from({length: qRewards.length}, (_, i) => i + 1);
    
    // Smooth the rewards for better visualization
    const smooth = (data, windowSize = 10) => {
        const smoothed = [];
        for (let i = 0; i < data.length; i++) {
            let sum = 0;
            let count = 0;
            for (let j = Math.max(0, i - windowSize + 1); j <= i; j++) {
                sum += data[j];
                count++;
            }
            smoothed.push(sum / count);
        }
        return smoothed;
    };

    rewardChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Q-Learning (Smoothed)',
                    data: smooth(qRewards),
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                },
                {
                    label: 'SARSA (Smoothed)',
                    data: smooth(sarsaRewards),
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f8fafc' }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Episodes', color: '#cbd5e1' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#cbd5e1' }
                },
                y: {
                    title: { display: true, text: 'Sum of Rewards during Episode', color: '#cbd5e1' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#cbd5e1' },
                    min: -100,
                    max: 0
                }
            }
        }
    });
}

function renderGrid(containerId, policy) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    const rows = 4;
    const cols = 12;
    const arrows = ['↑', '→', '↓', '←']; // 0: Up, 1: Right, 2: Down, 3: Left
    
    // Trace path based on optimal policy to highlight it
    let r = 3, c = 0;
    const path = new Set();
    path.add(`${r},${c}`);
    let steps = 0;
    
    while (steps < 50) { // Limit steps to prevent infinite loop
        let action = policy[r][c];
        if (action === 0) r = Math.max(0, r - 1);
        else if (action === 1) c = Math.min(cols - 1, c + 1);
        else if (action === 2) r = Math.min(rows - 1, r + 1);
        else if (action === 3) c = Math.max(0, c - 1);
        
        if (r === 3 && c >= 1 && c <= 10) { // Cliff
            break;
        }
        
        path.add(`${r},${c}`);
        
        if (r === 3 && c === 11) { // Goal
            break;
        }
        steps++;
    }
    
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            
            if (i === 3 && j === 0) cell.classList.add('start');
            else if (i === 3 && j === 11) cell.classList.add('goal');
            else if (i === 3 && j >= 1 && j <= 10) cell.classList.add('cliff');
            
            if (path.has(`${i},${j}`)) {
                cell.classList.add('path');
            }
            
            // Only draw arrows on non-terminal states
            if (!(i === 3 && j >= 1 && j <= 10) && !(i === 3 && j === 11)) {
                cell.innerText = arrows[policy[i][j]];
            }
            
            container.appendChild(cell);
        }
    }
}
