let chartInstance = null;
let lastResult    = null;
let lastText      = null;

// Character counter
document.getElementById("newsText").addEventListener("input", function () {
  document.getElementById("charCount").textContent = this.value.length;
});

async function analyze() {
  const text = document.getElementById("newsText").value.trim();
  const btn  = document.getElementById("analyzeBtn");

  document.getElementById("resultCard").classList.add("hidden");
  document.getElementById("errorCard").classList.add("hidden");

  if (text.length < 10) {
    showError("Please enter at least 10 characters.");
    return;
  }

  btn.disabled    = true;
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
    btn.disabled    = false;
    btn.textContent = "Analyze Text";
  }
}

function showResult(data) {
  lastResult = data;
  lastText   = document.getElementById("newsText").value.trim();

  const card       = document.getElementById("resultCard");
  const verdict    = document.getElementById("verdict");
  const confidence = document.getElementById("confidenceText");

  const icons = {
    FAKE:      "🔴 FAKE NEWS",
    REAL:      "🟢 REAL NEWS",
    UNCERTAIN: "🟡 UNCERTAIN"
  };
  verdict.textContent = icons[data.label];
  verdict.className   = `verdict ${data.label.toLowerCase()}`;
  confidence.textContent =
    `${data.confidence}% confident · Threshold: ${data.threshold * 100}%`;

  document.getElementById("realBar").style.width = data.real_prob + "%";
  document.getElementById("fakeBar").style.width = data.fake_prob + "%";
  document.getElementById("realPct").textContent = data.real_prob + "%";
  document.getElementById("fakePct").textContent = data.fake_prob + "%";

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

  // Reset feedback and fact-check UI
  document.getElementById("correctionDiv").classList.add("hidden");
  document.getElementById("feedbackMsg").classList.add("hidden");
  document.getElementById("factCheckResults").innerHTML = "";

  card.classList.remove("hidden");
  card.scrollIntoView({ behavior: "smooth" });
}

function showError(msg) {
  document.getElementById("errorMsg").textContent = msg;
  document.getElementById("errorCard").classList.remove("hidden");
}

function reset() {
  document.getElementById("newsText").value        = "";
  document.getElementById("charCount").textContent = "0";
  document.getElementById("resultCard").classList.add("hidden");
  document.getElementById("errorCard").classList.add("hidden");
  document.getElementById("factCheckResults").innerHTML = "";
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
  lastResult = null;
  lastText   = null;
}

// ── Feedback ─────────────────────────────────────────────
function sendFeedback(type) {
  if (type === "correct") {
    document.getElementById("feedbackMsg").textContent = "✅ Thanks for confirming!";
    document.getElementById("feedbackMsg").classList.remove("hidden");
    document.getElementById("correctionDiv").classList.add("hidden");

    fetch("/feedback", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text:            lastText,
        predicted_label: lastResult.label,
        correct_label:   lastResult.label,
        confidence:      lastResult.confidence
      })
    });

  } else {
    document.getElementById("correctionDiv").classList.remove("hidden");
    document.getElementById("feedbackMsg").classList.add("hidden");
  }
}

function submitCorrection(correctLabel) {
  fetch("/feedback", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text:            lastText,
      predicted_label: lastResult.label,
      correct_label:   correctLabel,
      confidence:      lastResult.confidence
    })
  }).then(() => {
    document.getElementById("feedbackMsg").textContent =
      "✅ Correction saved! This will help improve the model.";
    document.getElementById("feedbackMsg").classList.remove("hidden");
    document.getElementById("correctionDiv").classList.add("hidden");
  });
}

// ── Fact Check ────────────────────────────────────────────
async function runFactCheck() {
  const btn       = document.querySelector(".factcheck-btn");
  const container = document.getElementById("factCheckResults");

  btn.disabled    = true;
  btn.textContent = "Searching...";

  try {
    const res  = await fetch("/fact-check", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text: lastText })
    });
    const data = await res.json();

    if (!data.results || data.results.length === 0) {
      container.innerHTML = `
        <div class="fc-result">
          <span style="color:#64748b">
            No fact-checks or related news found for this claim.
          </span>
        </div>`;
    } else {
      // Source banner
      const sourceBanner = data.source === "news"
        ? `<div style="color:#f59e0b;font-size:0.8rem;
                       margin-bottom:8px;text-align:center">
               ⚠️ No direct fact-checks found —
               showing related news instead
           </div>`
        : `<div style="color:#22c55e;font-size:0.8rem;
                       margin-bottom:8px;text-align:center">
               ✅ Direct fact-checks found
           </div>`;

      container.innerHTML = sourceBanner + data.results.map(r => `
        <div class="fc-result">
          <div class="fc-rating">
            ${r.type === "news" ? "📰" : "🔍"} ${r.rating}
          </div>
          <div>
            <strong>${r.type === "news" ? "Related:" : "Claim:"}</strong>
            ${r.claim}
          </div>
          <div>
            <strong>By:</strong> ${r.claimant} ·
            <strong>Source:</strong> ${r.publisher}
          </div>
          ${r.url
            ? `<a href="${r.url}" target="_blank"
                  class="fc-link">Read more →</a>`
            : ""}
        </div>
      `).join("");
    }

  } catch (err) {
    container.innerHTML = `
      <div class="fc-result" style="color:#ef4444">
        Could not reach fact-check service.
      </div>`;
  } finally {
    btn.disabled    = false;
    btn.textContent = "🔎 Search Fact-Check Database";
  }
}