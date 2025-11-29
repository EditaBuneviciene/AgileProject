const API_BASE = 'http://localhost:5000/api';

// Global variables
let allYears = [];

// Load data from Python backend
async function loadDataFromPython() {
  try {
    console.log("Loading data from Python backend...");

    const response = await fetch(`${API_BASE}/load-data`);
    const result = await response.json();

    if (result.status === 'success') {
      console.log("Data loaded from Python backend");

      return {
        years: result.years || [],
        indicators: result.series || [],
        countries: result.countries || [],
      };
    }
    throw new Error(result.message);
  } catch (error) {
    console.error("Failed to load from Python backend:", error);
    document.getElementById("insightsContent").innerHTML =
      `<p style="color: red;">Error: Cannot connect to Python backend.</p>`;
    return { years: [], indicators: [], countries: [] };
  }
}

// Initialize dashboard
async function initializeDashboard() {
  await checkDataStatus();
  const data = await loadDataFromPython();

  // Store all years for later use
  allYears = data.years.map(item => item.value || item.name || item).sort();

  populateDropdowns(data);
  setupEventListeners();

  return data;
}

// Populate all dropdowns
function populateDropdowns(data) {
  const indicatorSel = document.getElementById("indicatorSelect");
  const countryASel = document.getElementById("countryA");
  const countryBSel = document.getElementById("countryB");
  const startYearSel = document.getElementById("startYear");
  const endYearSel = document.getElementById("endYear");

  // Clear existing options
  [indicatorSel, countryASel, countryBSel, startYearSel, endYearSel].forEach(sel => {
    sel.innerHTML = sel.id.includes('Year') ? '<option value="">-- Select --</option>' : `<option value="">-- Select ${sel.id.includes('country') ? 'Country' : 'Indicator'} --</option>`;
  });

  // Populate indicators
  data.indicators.forEach(item => {
    const name = item.name || item.value || item;
    const value = item.value || item.name || item;
    indicatorSel.add(new Option(name, value));
  });

  // Populate countries
  data.countries.forEach(item => {
    const name = item.name || item.value || item;
    const value = item.value || item.name || item;
    countryASel.add(new Option(name, value));
    countryBSel.add(new Option(name, value));
  });

  // Populate year ranges
  allYears.forEach(year => {
    startYearSel.add(new Option(year, year));
    endYearSel.add(new Option(year, year));
  });

  // Set default year range (first and last year)
  if (allYears.length > 0) {
    startYearSel.value = allYears[0];
    endYearSel.value = allYears[allYears.length - 1];
  }
}

// Setup event listeners
function setupEventListeners() {
  const generateBtn = document.getElementById("generateBtn");
  const enablePredictions = document.getElementById("enablePredictions");
  const predictionOptions = document.getElementById("predictionOptions");

  // Generate button click
  generateBtn.addEventListener("click", generateAnalysis);

  // Toggle prediction options
  enablePredictions.addEventListener("change", function() {
    predictionOptions.style.display = this.checked ? 'block' : 'none';
  });

  // Auto-update end year when start year changes
  document.getElementById("startYear").addEventListener("change", function() {
    const startYear = parseInt(this.value);
    const endYearSel = document.getElementById("endYear");

    if (startYear) {
      // Filter end years to be >= start year
      Array.from(endYearSel.options).forEach(option => {
        if (option.value && parseInt(option.value) < startYear) {
          option.disabled = true;
        } else {
          option.disabled = false;
        }
      });
    }
  });
}

// Main analysis function
async function generateAnalysis() {
  const indicator = document.getElementById("indicatorSelect").value;
  const countryA = document.getElementById("countryA").value;
  const countryB = document.getElementById("countryB").value;
  const startYear = document.getElementById("startYear").value;
  const endYear = document.getElementById("endYear").value;
  const chartType = document.querySelector('input[name="chartType"]:checked').value;
  const enablePredictions = document.getElementById("enablePredictions").checked;
  const predictionYears = enablePredictions ?
    document.querySelector('input[name="predictionYears"]:checked')?.value : null;

  // Validation
  if (!indicator) {
    showError("Please select an indicator");
    return;
  }

  const countriesToShow = [countryA, countryB].filter(c => c);
  if (countriesToShow.length === 0) {
    showError("Please select at least one country");
    return;
  }

  if (!startYear || !endYear) {
    showError("Please select start and end years");
    return;
  }

  // Generate year range
  const yearsToShow = generateYearRange(parseInt(startYear), parseInt(endYear));

  // Show loading state
  document.getElementById("insightsContent").innerHTML =
    "<p>Generating analysis... This may take a moment.</p>";

  // Generate chart
  await generateChartWithPython(indicator, countriesToShow, yearsToShow, chartType);

  // Generate insights and predictions
  await generateInsightsAndPredictions(indicator, countriesToShow, yearsToShow, enablePredictions, predictionYears);
}

// Generate year range array
function generateYearRange(start, end) {
  const years = [];
  for (let year = start; year <= end; year++) {
    years.push(year.toString());
  }
  return years;
}

// Show error message
function showError(message) {
  document.getElementById("insightsContent").innerHTML =
    `<div class="message error">${message}</div>`;
}

// Generate chart using Python backend
async function generateChartWithPython(series, countries, years, chartType = 'line') {
  try {
    console.log("Requesting chart from Python:", { series, countries, years, chartType });

    const response = await fetch(`${API_BASE}/generate-chart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        series: series,
        countries: countries,
        years: years,
        chart_type: chartType
      })
    });

    const result = await response.json();

    if (result.status === 'success') {
      const chartContainer = document.getElementById("chartContainer");
      chartContainer.innerHTML = `
        <div class="python-chart">
          <img src="${result.chart_image}" alt="Generated Chart" style="max-width: 100%; height: auto;">
          <p class="chart-source">📊 ${series} | ${years[0]} - ${years[years.length-1]} | ${countries.join(', ')}</p>
        </div>
      `;
      return result.country_data;
    } else {
      showError("Chart generation failed: " + result.message);
    }
  } catch (error) {
    console.log("Chart generation failed:", error);
    showError("Chart generation failed. Check Python backend.");
  }
  return null;
}

// Get AI insights from Python
async function getAIInsights(indicator, countries, years) {
  try {
    console.log("Requesting AI insights for:", { indicator, countries, years });

    const response = await fetch(`${API_BASE}/get-insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        series: indicator,
        countries: countries,
        years: years
      })
    });

    const result = await response.json();
    console.log("AI insights response:", result);

    return result;

  } catch (error) {
    console.error("AI insights error:", error);
    return { status: 'error', message: error.message };
  }
}

// Get predictions from Python
async function getPredictions(indicator, countries, years, predictionYears) {
  try {
    console.log("Requesting predictions:", { indicator, countries, years, predictionYears });

    const response = await fetch(`${API_BASE}/get-predictions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        series: indicator,
        countries: countries,
        years: years,
        prediction_years: predictionYears
      })
    });

    const result = await response.json();
    console.log("Predictions Response:", result);

    return result;

  } catch (error) {
    console.error("Predictions error:", error);
    return { status: 'error', message: error.message };
  }
}

// Generate insights and predictions
async function generateInsightsAndPredictions(indicator, countries, years, enablePredictions, predictionYears) {
  let insightsHTML = '<div class="insights-container">';

  try {
    // Get AI insights
    document.getElementById("insightsContent").innerHTML = "<p>Generating AI insights...</p>";

    const insightsResponse = await getAIInsights(indicator, countries, years);

    if (insightsResponse && insightsResponse.status === 'success') {
      insightsHTML += `
        <div class="insight-item">
          <h4>🤖 AI Analysis</h4>
          <div class="insight-text">${insightsResponse.insights}</div>
        </div>
      `;
    } else {
      insightsHTML += `
        <div class="insight-item">
          <h4>🤖 AI Analysis</h4>
          <div class="insight-text" style="color: #6c757d; font-style: italic;">
            ${insightsResponse?.message || 'AI insights currently unavailable.'}
          </div>
        </div>
      `;
    }

    // Update UI with insights first
    document.getElementById("insightsContent").innerHTML = insightsHTML;

    // Get predictions if enabled
    if (enablePredictions && predictionYears) {
      document.getElementById("insightsContent").innerHTML += `<p>Generating ${predictionYears}-year predictions...</p>`;

      const predictionsResponse = await getPredictions(indicator, countries, years, parseInt(predictionYears));

      if (predictionsResponse && predictionsResponse.status === 'success') {
        insightsHTML += `
          <div class="insight-item prediction-results">
            <h4>📈 ${predictionYears}-Year Predictions</h4>
            <div class="prediction-text">${formatPredictions(predictionsResponse.predictions)}</div>
          </div>
        `;
      } else {
        insightsHTML += `
          <div class="insight-item">
            <h4>📈 Predictions</h4>
            <div class="prediction-text" style="color: #6c757d;">
              ${predictionsResponse?.message || 'Predictions are currently unavailable.'}
            </div>
          </div>
        `;
      }
    }

    insightsHTML += '</div>';
    document.getElementById("insightsContent").innerHTML = insightsHTML;

  } catch (error) {
    console.error("Analysis error:", error);
    document.getElementById("insightsContent").innerHTML =
      `<div class="message error">Analysis failed: ${error.message}</div>`;
  }
}

// Format predictions for display
function formatPredictions(predictions) {
  if (!predictions || !predictions.predictions) return "No predictions available.";

  let html = "";
  for (const [country, values] of Object.entries(predictions.predictions)) {
    html += `<p><strong>${country}:</strong> ${values.join(', ')}</p>`;
  }
  return html;
}

// Data Management Functions
async function checkDataStatus() {
    const statusElement = document.getElementById('dataStatus');
    const messageElement = document.getElementById('uploadMessage');

    statusElement.textContent = 'Checking...';
    statusElement.className = 'status-checking';
    messageElement.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE}/check-data-status`);
        const result = await response.json();

        if (result.status === 'success') {
            statusElement.textContent = result.data_available ? 'Data Available' : 'No Data Found';
            statusElement.className = result.data_available ? 'status-available' : 'status-missing';

            if (!result.data_available) {
                messageElement.innerHTML = '<div class="message info">Please upload a CSV file to get started</div>';
            }
        }
    } catch (error) {
        statusElement.textContent = 'Check Failed';
        statusElement.className = 'status-missing';
        messageElement.innerHTML = `<div class="message error">Unable to check data status: ${error.message}</div>`;
    }
}

async function uploadCSV() {
    const fileInput = document.getElementById('csvUpload');
    const messageElement = document.getElementById('uploadMessage');

    if (!fileInput.files.length) {
        messageElement.innerHTML = '<div class="message error">Please select a CSV file first</div>';
        return;
    }

    const file = fileInput.files[0];
    if (!file.name.toLowerCase().endsWith('.csv')) {
        messageElement.innerHTML = '<div class="message error">Please select a CSV file</div>';
        return;
    }

    messageElement.innerHTML = '<div class="message info">Uploading and processing file... This may take a moment.</div>';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/upload-csv`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.status === 'success') {
            messageElement.innerHTML = `<div class="message success">${result.message}</div>`;
            setTimeout(() => {
                checkDataStatus();
                location.reload();
            }, 2000);
        } else {
            messageElement.innerHTML = `<div class="message error">Upload failed: ${result.message}</div>`;
        }
    } catch (error) {
        messageElement.innerHTML = `<div class="message error">Upload error: ${error.message}</div>`;
    }
}

// Initialize the dashboard
initializeDashboard();