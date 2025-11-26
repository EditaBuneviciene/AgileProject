const API_BASE = 'http://localhost:5000/api';

async function loadData() {
  const response = await fetch("../data/Data_Cleaned.csv");
  const text = await response.text();

  // Auto-detect delimiter
  const delimiter = text.includes(";") ? ";" : ",";
  const rows = text.trim().split(/\r?\n/);
  const headers = rows[0].split(delimiter);
  const years = headers.slice(2).map(h => h.trim());

  const dataByIndicator = {};
  const indicators = new Set();
  const countries = new Set();

  for (let i = 1; i < rows.length; i++) {
    const cols = rows[i].split(delimiter);
    if (cols.length < 3) continue;
    const indicator = cols[0].trim();
    const country = cols[1].trim();
    const values = cols.slice(2).map(v => parseFloat(v) || 0);

    indicators.add(indicator);
    countries.add(country);

    if (!dataByIndicator[indicator]) dataByIndicator[indicator] = {};
    dataByIndicator[indicator][country] = values;
  }

  console.log("Loaded years:", years);
  console.log("Indicators:", indicators.size, "Countries:", countries.size);

  return { years, dataByIndicator, indicators: [...indicators].sort(), countries: [...countries].sort() };
}

function calculateYoY(data) {
  const result = [0];
  for (let i = 1; i < data.length; i++) {
    const prev = data[i - 1];
    const curr = data[i];
    result.push(prev ? ((curr - prev) / prev) * 100 : 0);
  }
  return result;
}

function generateSummary(countryA, countryB, indicator, data) {
  const set = data[indicator];
  if (!set) return;
  let html = `<h3>${indicator}</h3>`;
  [countryA, countryB].forEach(c => {
    if (c && set[c]) {
      const vals = set[c];
      const latest = vals[vals.length - 1];
      const prev = vals[vals.length - 2];
      const change = ((latest - prev) / prev * 100).toFixed(2);
      html += `<p><b>${c}</b>: ${latest.toFixed(2)} (${change > 0 ? "+" : ""}${change}% YoY)</p>`;
    }
  });
  document.getElementById("summaryContent").innerHTML = html;
}

// Simple Python API integration
async function connectToPythonBackend() {
  try {
    const response = await fetch(`${API_BASE}/load-data`);
    const result = await response.json();
    
    if (result.status === 'success') {
      console.log("Connected to Python backend");
      return result;
    }
  } catch (error) {
    console.log("Python backend not available, using CSV data only");
  }
  return null;
}

// Get AI insights from Python
async function getAIInsights(indicator, countryA, countryB) {
  try {
    const response = await fetch(`${API_BASE}/get-insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        series: indicator,
        countries: [countryA, countryB].filter(c => c),
        years: [] // Python will handle this
      })
    });
    
    const result = await response.json();
    return result.status === 'success' ? result.insights : null;
  } catch (error) {
    return null;
  }
}

// Add AI features to summary
function enhanceSummaryWithAI(summaryElement, indicator, countryA, countryB) {
  const aiSection = document.createElement('div');
  aiSection.className = 'ai-features';
  aiSection.innerHTML = `
    <h4>🤖 AI Analysis</h4>
    <button onclick="showAIInsights('${indicator}', '${countryA}', '${countryB}')" class="ai-btn">
      Get AI Insights
    </button>
  `;
  summaryElement.appendChild(aiSection);
}

// Display AI insights
async function showAIInsights(indicator, countryA, countryB) {
  const insights = await getAIInsights(indicator, countryA, countryB);
  const summaryContent = document.getElementById("summaryContent");
  
  if (insights) {
    summaryContent.innerHTML = `
      <div class="ai-insights">
        <h3>🤖 AI Analysis: ${indicator}</h3>
        <div class="insight-text">${insights}</div>
        <button onclick="updateChart()" class="back-btn">← Back to Chart</button>
      </div>
    `;
  } else {
    summaryContent.innerHTML += `<p class="ai-error">AI insights currently unavailable</p>`;
  }
}

// Enhanced generateSummary to include AI features
function generateSummary(countryA, countryB, indicator, data) {
  const set = data[indicator];
  if (!set) return;
  
  let html = `<h3>${indicator}</h3>`;
  [countryA, countryB].forEach(c => {
    if (c && set[c]) {
      const vals = set[c];
      const latest = vals[vals.length - 1];
      const prev = vals[vals.length - 2];
      const change = ((latest - prev) / prev * 100).toFixed(2);
      html += `<p><b>${c}</b>: ${latest.toFixed(2)} (${change > 0 ? "+" : ""}${change}% YoY)</p>`;
    }
  });
  
  document.getElementById("summaryContent").innerHTML = html;
  
  // Add AI features if valid data
  if (indicator && (countryA || countryB)) {
    const summaryElement = document.getElementById("summaryContent");
    enhanceSummaryWithAI(summaryElement, indicator, countryA, countryB);
  }
}

loadData().then(({ years, dataByIndicator, indicators, countries }) => {
  const indicatorSel = document.getElementById("indicatorSelect");
  const countryASel = document.getElementById("countryA");
  const countryBSel = document.getElementById("countryB");
  const yearSel = document.getElementById("yearSelect");
  const showChange = document.getElementById("showChange");
  const ctx = document.getElementById("trendChart").getContext("2d");

  indicators.forEach(i => indicatorSel.add(new Option(i, i)));
  countries.forEach(c => {
    countryASel.add(new Option(c, c));
    countryBSel.add(new Option(c, c));
  });
  years.forEach(y => yearSel.add(new Option(y, y)));

  const chart = new Chart(ctx, {
    type: "line",
    data: { labels: years, datasets: [] },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: { y: { beginAtZero: false } }
    }
  });

  function updateChart() {
    const ind = indicatorSel.value;
    const dataset = dataByIndicator[ind];
    if (!dataset) return;
    const cA = countryASel.value;
    const cB = countryBSel.value;
    const selectedYear = yearSel.value;
    const yoY = showChange.checked;
    const colors = ["#e63946", "#1d3557"];
    const sets = [];

    [cA, cB].forEach((c, i) => {
      if (c && dataset[c]) {
        let data = dataset[c];
        if (selectedYear) {
          const idx = years.indexOf(selectedYear);
          data = idx !== -1 ? [data[idx]] : [];
        }
        sets.push({
          label: c + (selectedYear ? ` (${selectedYear})` : ""),
          data: yoY ? calculateYoY(dataset[c]) : data,
          borderColor: colors[i],
          backgroundColor: colors[i] + "88",
          borderWidth: 2,
          fill: false
        });
      }
    });

    chart.data.labels = selectedYear ? [selectedYear] : years;
    chart.data.datasets = sets;
    chart.config.type = selectedYear ? "bar" : "line";
    chart.update();

    generateSummary(cA, cB, ind, dataByIndicator);
  }

  [indicatorSel, countryASel, countryBSel, yearSel, showChange].forEach(el =>
    el.addEventListener("change", updateChart)
  );

  // Try to connect to Python backend on startup
  connectToPythonBackend().then(pythonData => {
    if (pythonData) {
      console.log("Python backend connected successfully!");
    }
  });
});