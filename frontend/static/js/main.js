let chartInstance = null;

// Character counter
document.getElementById("newsText").addEventListener("input", function () {
  document.getElementById("charCount").textContent = this.value.length;
});

async function analyze() {
  const text = document.getElementById("newsText").value.trim();
  const btn  = document.getElementById("analyzeBtn");

  // Hide previous results
  document.getElementById("resultCard").classList.add("hidden");
  document.getElementById("errorCard").classList.add("hidden");

  if (text.length < 10) {
    showError("Please enter at least 10 characters.");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Analyzing...";

  try {
    const response = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text })
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || "Something went wrong.");
      return;
    }

    showResult(data);

  } catch (err) {
    showError("Could not connect to API. Make sure the Flask server is running.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze Text";
  }
}

function showResult(data) {
  const card       = document.getElementById("resultCard");
  const verdict    = document.getElementById("verdict");
  const confidence = document.getElementById("confidenceText");

  // Verdict
  const icons = { FAKE: "🔴 FAKE NEWS", REAL: "🟢 REAL NEWS", UNCERTAIN: "🟡 UNCERTAIN" };
  verdict.textContent = icons[data.label];
  verdict.className   = `verdict ${data.label.toLowerCase()}`;
  confidence.textContent = `${data.confidence}% confident · Threshold: ${data.threshold * 100}%`;

  // Probability bars
  document.getElementById("realBar").style.width = data.real_prob + "%";
  document.getElementById("fakeBar").style.width = data.fake_prob + "%";
  document.getElementById("realPct").textContent  = data.real_prob + "%";
  document.getElementById("fakePct").textContent  = data.fake_prob + "%";

  // Doughnut chart
  if (chartInstance) chartInstance.destroy();
  const ctx = document.getElementById("probChart").getContext("2d");
  chartInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels:   ["REAL", "FAKE"],
      datasets: [{
        data:            [data.real_prob, data.fake_prob],
        backgroundColor: ["#22c55e", "#ef4444"],
        borderWidth:     0
      }]
    },
    options: {
      cutout: "70%",
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { size: 13 } }
        }
      }
    }
  });

  card.classList.remove("hidden");
  card.scrollIntoView({ behavior: "smooth" });
}

function showError(msg) {
  document.getElementById("errorMsg").textContent  = msg;
  document.getElementById("errorCard").classList.remove("hidden");
}

function reset() {
  document.getElementById("newsText").value = "";
  document.getElementById("charCount").textContent = "0";
  document.getElementById("resultCard").classList.add("hidden");
  document.getElementById("errorCard").classList.add("hidden");
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
}