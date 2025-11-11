document.addEventListener("DOMContentLoaded", () => {
  const ctx = document.getElementById("trendChart").getContext("2d");

  // Create empty chart
  const chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: []
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        title: { display: true, text: "No Data Loaded" }
      },
      scales: { y: { beginAtZero: false } }
    }
  });

  // Dropdown and checkbox event placeholders
  const indicator = document.getElementById("indicatorSelect");
  const countryA = document.getElementById("countryA");
  const countryB = document.getElementById("countryB");
  const whereBy = document.getElementById("whereBy");
  const showChange = document.getElementById("showChange");

  [indicator, countryA, countryB, whereBy, showChange].forEach(el => {
    el.addEventListener("change", () => {
      chart.options.plugins.title.text = "Selections changed, but no data connected";
      chart.update();
      document.getElementById("summaryContent").innerHTML =
        "<p>Dataset connection pending...</p>";
    });
  });
});
