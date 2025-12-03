const API_BASE = 'http://localhost:5000/api';

// Global variables
let allYears = [];
let currentChartData = null;

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
      `<div class="message error">Error: Cannot connect to Python backend. Make sure Flask server is running on port 5000.</div>`;
    return { years: [], indicators: [], countries: [] };
  }
}

// Initialize dashboard
async function initializeDashboard() {
  await checkDataStatus();
  const data = await loadDataFromPython();

  // Store all years for later use
  allYears = data.years.map(item => String(item.value || item.name || item)).sort();

  populateDropdowns(data);
  setupEventListeners();

  return data;
}

// Populate dropdowns
function populateDropdowns(data) {
  const indicatorSel = document.getElementById("indicatorSelect");
  const countryASel = document.getElementById("countryA");
  const countryBSel = document.getElementById("countryB");
  const startYearSel = document.getElementById("startYear");
  const endYearSel = document.getElementById("endYear");

  [indicatorSel, countryASel, countryBSel, startYearSel, endYearSel].forEach(sel => {
    sel.innerHTML = sel.id.includes('Year')
      ? '<option value="">-- Select --</option>'
      : `<option value="">-- Select ${sel.id.includes('country') ? 'Country' : 'Indicator'} --</option>`;
  });

  // Indicators
  data.indicators.forEach(item => {
    const name = item.name || item.value || item;
    const value = item.value || item.name || item;
    indicatorSel.add(new Option(name, value));
  });

  // Countries
  data.countries.forEach(item => {
    const name = item.name || item.value || item;
    const value = item.value || item.name || item;
    countryASel.add(new Option(name, value));
    countryBSel.add(new Option(name, value));
  });

  // Years
  allYears.forEach(year => {
    startYearSel.add(new Option(year, year));
    endYearSel.add(new Option(year, year));
  });

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

  generateBtn.addEventListener("click", generateAnalysis);

  enablePredictions.addEventListener("change", function () {
    predictionOptions.style.display = this.checked ? 'block' : 'none';

    if (this.checked &&
      !document.querySelector('input[name="predictionYears"]:checked')) {
      document.querySelector('input[name="predictionYears"][value="5"]').checked = true;
    }
  });

  document.getElementById("startYear").addEventListener("change", function () {
    const startYear = parseInt(this.value);
    const endYearSel = document.getElementById("endYear");

    if (startYear && endYearSel.value && parseInt(endYearSel.value) < startYear) {
      endYearSel.value = startYear;
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
  const predictionYearsChoice = document.querySelector('input[name="predictionYears"]:checked');
  const predictionYears = enablePredictions ? (predictionYearsChoice ? predictionYearsChoice.value : null) : null;

  if (!indicator) return showError("Please select an indicator");

  const countriesToShow = [countryA, countryB].filter(c => c);
  if (countriesToShow.length === 0) return showError("Please select at least one country");

  if (!startYear || !endYear) return showError("Please select start and end years");

  // Generate year range
  const yearsToShow = generateYearRange(parseInt(startYear), parseInt(endYear));

  showChartLoading();
  document.getElementById("insightsContent").innerHTML =
    "<div class='message info'>Generating analysis... This may take a moment.</div>";

  try {
    currentChartData = await generateChartWithPython(
      indicator,
      countriesToShow,
      yearsToShow,
      chartType
    );

    if (currentChartData) {
      await generateInsightsAndPredictions(
        indicator,
        countriesToShow,
        yearsToShow,
        enablePredictions,
        predictionYears
      );
    }

  } catch (error) {
    console.error("Analysis error:", error);
    showError("Analysis failed: " + error.message);
  }
}

function showChartLoading() {
  const chartContainer = document.getElementById("chartContainer");
  chartContainer.innerHTML = `
    <div class="chart-loading">
      <p>Generating chart...</p>
    </div>
  `;
}

function generateYearRange(start, end) {
  const years = [];
  for (let y = start; y <= end; y++) years.push(String(y));
  return years;
}

function showError(message) {
  document.getElementById("insightsContent").innerHTML =
    `<div class="message error">${message}</div>`;
}

// Generate chart
async function generateChartWithPython(series, countries, years, chartType = 'line',
  includePredictions = false, predictions = null, predictionYears = 0) {

  try {
    console.log("Requesting chart from Python:", { series, countries, years, chartType });

    const requestBody = {
      series: series,
      countries: countries,
      years: years,
      chart_type: chartType,
      include_predictions: includePredictions,
      prediction_years: parseInt(predictionYears) || 0
    };

    if (includePredictions && predictions) {
      requestBody.predictions = predictions;
    }

    const response = await fetch(`${API_BASE}/generate-chart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    const result = await response.json();
    console.log("Chart response:", result);

    if (result.status === 'success') {
      const chartContainer = document.getElementById("chartContainer");
      chartContainer.innerHTML = `
        <div class="python-chart">
          <img src="${result.chart_image}" alt="Generated Chart"
               style="max-width: 100%; height: auto; border-radius: 8px;">
          <p class="chart-source">📊 ${series} | ${years[0]} - ${years[years.length - 1]} | ${countries.join(', ')}</p>
        </div>
      `;
      return result.country_data;
    }

    showError("Chart generation failed: " + result.message);
    return null;

  } catch (error) {
    console.error("Chart generation failed:", error);
    showError("Chart generation failed. Please check backend.");
    return null;
  }
}

// Ai Insights
async function getAIInsights(indicator, countries, years) {
  try {
    const response = await fetch(`${API_BASE}/get-insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ series: indicator, countries, years })
    });

    return await response.json();
  } catch (error) {
    return { status: 'error', message: error.message };
  }
}

// Predictions
async function getPredictions(indicator, countries, years, predictionYears) {
  try {
    const response = await fetch(`${API_BASE}/get-predictions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        series: indicator,
        countries,
        years,
        prediction_years: parseInt(predictionYears)
      })
    });

    return await response.json();
  } catch (error) {
    return { status: 'error', message: error.message };
  }
}

// Insights report plus prediction for insight reporting
async function generateInsightsAndPredictions(indicator, countries, years,
  enablePredictions, predictionYears) {

  let insightsHTML = '<div class="insights-container">';

  try {
    document.getElementById("insightsContent").innerHTML =
      "<div class='message info'>Generating AI insights...</div>";

    const insightsResponse = await getAIInsights(indicator, countries, years);

    if (insightsResponse.status === 'success') {
      insightsHTML += `
        <div class="insight-item">
          <h4>🤖 AI Analysis</h4>
          <div class="insight-text">${formatInsights(insightsResponse.insights)}</div>
        </div>`;
    } else {
      insightsHTML += `
        <div class="insight-item">
          <h4>🤖 AI Analysis</h4>
          <div class="insight-text" style="color:#6c757d;">
            ${insightsResponse.message || 'AI unavailable.'}
          </div>
        </div>`;
    }

    document.getElementById("insightsContent").innerHTML = insightsHTML;

    // Predictions (if enabled)
    if (enablePredictions && predictionYears) {
      document.getElementById("insightsContent").innerHTML +=
        `<div class='message info'>Generating ${predictionYears}-year predictions...</div>`;

      const predictionsResponse = await getPredictions(
        indicator, countries, years, predictionYears
      );

      if (predictionsResponse.status === 'success') {

        // Regenerate chart with predictions added
        await generateChartWithPython(
          indicator,
          countries,
          years,
          document.querySelector('input[name="chartType"]:checked').value,
          true,
          predictionsResponse.predictions,
          parseInt(predictionYears)
        );

        insightsHTML += `
          <div class="insight-item prediction-results">
            <h4>📈 ${predictionYears}-Year Predictions</h4>
            <div class="prediction-text">
              ${formatPredictions(predictionsResponse.predictions, countries)}
            </div>
          </div>`;

      } else {
        insightsHTML += `
          <div class="insight-item">
            <h4>📈 Predictions</h4>
            <div class="prediction-text" style="color:#6c757d;">
              ${predictionsResponse.message || 'Predictions unavailable.'}
            </div>
          </div>`;
      }
    }

    insightsHTML += '</div>';
    document.getElementById("insightsContent").innerHTML = insightsHTML;

  } catch (error) {
    document.getElementById("insightsContent").innerHTML =
      `<div class="message error">Analysis failed: ${error.message}</div>`;
  }
}

function formatInsights(insights) {
  if (!insights) return "No insights available.";
  return insights.split('\n').map(line => `<p>${line}</p>`).join('');
}

function formatPredictions(predictions, countries) {
  if (!predictions || !predictions.predictions)
    return "<p>No predictions available.</p>";

  let html = "";
  for (const country of countries) {
    if (predictions.predictions[country]) {
      const vals = predictions.predictions[country];
      html += `
        <div style="margin-bottom:1rem;">
          <strong>${country}:</strong><br>
          ${vals.map((v, i) => `Year ${i + 1}: <strong>${v}</strong>`).join('<br>')}
        </div>`;
    }
  }

  return html;
}

// Data management
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
      if (result.data_available) {
        statusElement.textContent = 'Data Available ✅';
        statusElement.className = 'status-available';
      } else {
        statusElement.textContent = 'No Data Found ❌';
        statusElement.className = 'status-missing';
        messageElement.innerHTML =
          '<div class="message info">Please upload a CSV file.</div>';
      }
    }
  } catch (error) {
    statusElement.textContent = 'Check Failed ❌';
    statusElement.className = 'status-missing';
    messageElement.innerHTML =
      `<div class="message error">Unable to check data: ${error.message}</div>`;
  }
}

async function uploadCSV() {
  const fileInput = document.getElementById('csvUpload');
  const messageElement = document.getElementById('uploadMessage');

  if (!fileInput.files.length) {
    messageElement.innerHTML = '<div class="message error">Select a CSV first</div>';
    return;
  }

  const file = fileInput.files[0];
  if (!file.name.toLowerCase().endsWith('.csv')) {
    messageElement.innerHTML = '<div class="message error">File must be CSV</div>';
    return;
  }

  messageElement.innerHTML =
    '<div class="message info">Uploading... Please wait.</div>';

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/upload-csv`, {
      method: 'POST',
      body: formData
    });

    const result = await response.json();

    if (result.status === 'success') {
      messageElement.innerHTML =
        `<div class="message success">${result.message}</div>`;
      setTimeout(() => {
        checkDataStatus();
        initializeDashboard();
      }, 2000);
    } else {
      messageElement.innerHTML =
        `<div class="message error">Upload failed: ${result.message}</div>`;
    }
  } catch (error) {
    messageElement.innerHTML =
      `<div class="message error">Upload error: ${error.message}</div>`;
  }
}

function downloadDashboardPDF() {
  const { jsPDF } = window.jspdf;

  // Choose what part of the page to capture.
  // You can use document.body, or a specific wrapper if you prefer.
  const dashboardElement = document.body;

  // Optional: scroll to top so layout isn't mid-scroll
  window.scrollTo(0, 0);

  html2canvas(dashboardElement, { scale: 2 })
    .then(canvas => {
      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();

      const imgWidth = pageWidth;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      let heightLeft = imgHeight;
      let position = 0;

      // First page
      pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;

      // Extra pages if content is longer than one page
      while (heightLeft > 0) {
        position = heightLeft - imgHeight;
        pdf.addPage();
        pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
      }

      pdf.save("health-dashboard.pdf");
    })
    .catch(err => {
      console.error("PDF generation failed:", err);
      alert("Failed to generate PDF.");
    });
}

// Initialize page
document.addEventListener('DOMContentLoaded', initializeDashboard);
