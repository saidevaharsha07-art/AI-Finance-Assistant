const chartDataNode = document.getElementById("chart-data");

if (chartDataNode) {
  const charts = JSON.parse(chartDataNode.textContent);
  const colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#475569", "#65a30d", "#be123c", "#0d9488", "#9333ea"];

  new Chart(document.getElementById("incomeChart"), {
    type: "bar",
    data: {
      labels: charts.income_expenses.labels,
      datasets: [{
        label: "Amount",
        data: charts.income_expenses.values,
        backgroundColor: ["#2563eb", "#dc2626", "#16a34a"]
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } }
    }
  });

  new Chart(document.getElementById("budgetChart"), {
    type: "doughnut",
    data: {
      labels: charts.budget_allocation.labels.map((label) => label.replaceAll("_", " ")),
      datasets: [{
        data: charts.budget_allocation.values,
        backgroundColor: colors
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } }
    }
  });

  new Chart(document.getElementById("riskChart"), {
    type: "pie",
    data: {
      labels: charts.risk.labels,
      datasets: [{
        data: charts.risk.values,
        backgroundColor: ["#16a34a", "#dc2626"]
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } }
    }
  });

  new Chart(document.getElementById("forecastChart"), {
    type: "line",
    data: {
      labels: charts.savings_forecast.labels,
      datasets: [{
        label: "Cumulative savings",
        data: charts.savings_forecast.values,
        borderColor: "#16a34a",
        backgroundColor: "rgba(22, 163, 74, 0.12)",
        fill: true,
        tension: 0.35
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: false } }
    }
  });

  new Chart(document.getElementById("expenseTrendChart"), {
    type: "line",
    data: {
      labels: charts.expense_forecast.labels,
      datasets: [{
        label: "Predicted expenses",
        data: charts.expense_forecast.values,
        borderColor: "#dc2626",
        backgroundColor: "rgba(220, 38, 38, 0.10)",
        fill: true,
        tension: 0.35
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: false } }
    }
  });
}

const simulationButton = document.getElementById("runSimulation");
const financeForm = document.getElementById("financeForm");
const simulationResult = document.getElementById("simulationResult");

if (simulationButton && financeForm && simulationResult) {
  simulationButton.addEventListener("click", async () => {
    const data = Object.fromEntries(new FormData(financeForm).entries());
    data.monthly_income = Number(data.monthly_income || 0) + Number(document.getElementById("simIncome").value || 0);
    data.annual_income = Number(data.monthly_income || 0) * 12;
    data.emi_loans = Number(data.emi_loans || 0) + Number(document.getElementById("simEmi").value || 0);
    data.shopping = Number(data.shopping || 0) + Number(document.getElementById("simShopping").value || 0);
    data.savings_goal = Number(data.savings_goal || 0) + Number(document.getElementById("simGoal").value || 0);

    simulationResult.textContent = "Running simulation...";
    const response = await fetch("/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    const result = await response.json();
    simulationResult.innerHTML = `
      <span>Simulated health score</span>
      <strong>${result.health_score}/100 (${result.health_category})</strong>
      <small>Class: ${result.spender_class} | Savings: Rs. ${Math.round(result.actual_savings)} | 12-month forecast: Rs. ${Math.round(result.forecast.month_12)}</small>
    `;
  });
}
