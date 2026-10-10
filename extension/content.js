let popup        = null;
let currentData  = null;
let selectedText = null;

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "SHOW_LOADING") showPopup(null, null, true);
  if (msg.type === "SHOW_RESULT") {
    if (msg.error) showPopup(null, msg.error, false);
    else           showPopup(msg.data, null, false);
  }
});

function showPopup(data, error, loading) {
  if (popup) popup.remove();

  popup = document.createElement("div");
  popup.id = "fnd-popup";

  if (loading) {
    popup.innerHTML = `
      <div class="fnd-header">
        <span>🔍 Fake News Detector</span>
        <button id="fnd-close-btn">✕</button>
      </div>
      <div class="fnd-body">
        <div class="fnd-loading">Analyzing text...</div>
      </div>`;

  } else if (error) {
    popup.innerHTML = `
      <div class="fnd-header">
        <span>🔍 Fake News Detector</span>
        <button id="fnd-close-btn">✕</button>
      </div>
      <div class="fnd-body">
        <div class="fnd-error">${error}</div>
      </div>`;

  } else {
    currentData  = data;
    selectedText = data.input_text || "";

    const icons  = { FAKE: "🔴", REAL: "🟢", UNCERTAIN: "🟡" };
    const colors = {
      FAKE:      "#ef4444",
      REAL:      "#22c55e",
      UNCERTAIN: "#f59e0b"
    };

    popup.innerHTML = `
      <div class="fnd-header">
        <span>🔍 Fake News Detector</span>
        <button id="fnd-close-btn">✕</button>
      </div>
      <div class="fnd-body">

        <div class="fnd-verdict" style="color:${colors[data.label]}">
          ${icons[data.label]} ${data.label}
        </div>
        <div class="fnd-confidence">${data.confidence}% confident</div>

        <div class="fnd-bars">
          <div class="fnd-bar-row">
            <span>🟢 REAL</span>
            <div class="fnd-bar-bg">
              <div class="fnd-bar-fill"
                   style="width:${data.real_prob}%;background:#22c55e">
              </div>
            </div>
            <span>${data.real_prob}%</span>
          </div>
          <div class="fnd-bar-row">
            <span>🔴 FAKE</span>
            <div class="fnd-bar-bg">
              <div class="fnd-bar-fill"
                   style="width:${data.fake_prob}%;background:#ef4444">
              </div>
            </div>
            <span>${data.fake_prob}%</span>
          </div>
        </div>

        <div class="fnd-divider"></div>

        <div class="fnd-section-label">Was this correct?</div>
        <div class="fnd-feedback-btns">
          <button id="fnd-yes-btn" class="fnd-fb correct">👍 Yes</button>
          <button id="fnd-no-btn"  class="fnd-fb incorrect">👎 No</button>
        </div>
        <div class="fnd-feedback-msg" id="fnd-fb-msg"></div>

        <div class="fnd-correction" id="fnd-correction">
          <div class="fnd-section-label">What should it be?</div>
          <div class="fnd-correction-btns">
            <button id="fnd-correct-real">🟢 REAL</button>
            <button id="fnd-correct-fake">🔴 FAKE</button>
            <button id="fnd-correct-uncertain">🟡 UNCERTAIN</button>
          </div>
        </div>

        <div class="fnd-divider"></div>

        <button id="fnd-fc-btn" class="fnd-factcheck-btn">
          🔎 Search Fact-Check Database
        </button>
        <div id="fnd-fc-results"></div>

        <div class="fnd-scroll-hint" id="fnd-scroll-hint">
          ↓ scroll for more
        </div>

      </div>`;
  }

  document.body.appendChild(popup);

  // ── Scroll listener ───────────────────────────────────
  popup.addEventListener("scroll", function () {
    const hint = document.getElementById("fnd-scroll-hint");
    if (!hint) return;
    const atBottom =
      popup.scrollTop + popup.clientHeight >= popup.scrollHeight - 10;
    hint.style.opacity    = atBottom ? "0" : "1";
    hint.style.transition = "opacity 0.3s";
  });

  // ── Event listeners ───────────────────────────────────
  document.getElementById("fnd-close-btn")
    .addEventListener("click", () => popup.remove());

  if (!loading && !error) {
    document.getElementById("fnd-correction").style.display = "none";

    // Yes — correct prediction
    document.getElementById("fnd-yes-btn")
      .addEventListener("click", () => {
        document.getElementById("fnd-correction").style.display = "none";
        const msg = document.getElementById("fnd-fb-msg");
        msg.textContent = "✅ Thanks for confirming!";
        msg.style.color = "#22c55e";
        postFeedback(data.label, data.label, data.confidence);
      });

    // No — wrong prediction
    document.getElementById("fnd-no-btn")
      .addEventListener("click", () => {
        document.getElementById("fnd-correction").style.display = "block";
        document.getElementById("fnd-fb-msg").textContent = "";
      });

    // Correction buttons
    document.getElementById("fnd-correct-real")
      .addEventListener("click", () =>
        submitCorrection("REAL", data));
    document.getElementById("fnd-correct-fake")
      .addEventListener("click", () =>
        submitCorrection("FAKE", data));
    document.getElementById("fnd-correct-uncertain")
      .addEventListener("click", () =>
        submitCorrection("UNCERTAIN", data));

    // Fact check
    document.getElementById("fnd-fc-btn")
      .addEventListener("click", () => runFactCheck());
  }

  // Auto close after 30 seconds
  setTimeout(() => { if (popup) popup.remove(); }, 30000);
}

// ── Feedback ─────────────────────────────────────────────
// Routed through background.js to avoid mixed content block
function postFeedback(predictedLabel, correctLabel, confidence) {
  chrome.runtime.sendMessage({
    type:     "API_REQUEST",
    endpoint: "/feedback",
    body: {
      text:            selectedText,
      predicted_label: predictedLabel,
      correct_label:   correctLabel,
      confidence:      confidence
    }
  });
}

function submitCorrection(correctLabel, data) {
  postFeedback(data.label, correctLabel, data.confidence);
  document.getElementById("fnd-correction").style.display = "none";
  const msg = document.getElementById("fnd-fb-msg");
  msg.textContent = "✅ Correction saved!";
  msg.style.color = "#22c55e";
}

// ── Fact Check ───────────────────────────────────────────
// Routed through background.js to avoid mixed content block
function runFactCheck() {
  const btn     = document.getElementById("fnd-fc-btn");
  const results = document.getElementById("fnd-fc-results");

  btn.disabled    = true;
  btn.textContent = "Searching...";

  chrome.runtime.sendMessage(
    {
      type:     "API_REQUEST",
      endpoint: "/fact-check",
      body:     { text: selectedText }
    },
    (response) => {
      btn.disabled    = false;
      btn.textContent = "🔎 Search Fact-Check Database";

      if (!response || response.error) {
        results.innerHTML = `
          <div class="fnd-fc-item" style="color:#ef4444">
            Could not reach API. Make sure Flask is running.
          </div>`;
        return;
      }

      const data = response.data;

      if (!data.results || data.results.length === 0) {
        results.innerHTML = `
          <div class="fnd-fc-item">
            <span style="color:#64748b;font-size:0.75rem">
              No fact-checks or related news found.
            </span>
          </div>`;
      } else {
        const banner = data.source === "news"
          ? `<div style="color:#f59e0b;font-size:0.75rem;
                         margin-bottom:6px">
                 ⚠️ Showing related news (no direct fact-checks found)
             </div>`
          : `<div style="color:#22c55e;font-size:0.75rem;
                         margin-bottom:6px">
                 ✅ Direct fact-checks found
             </div>`;

        results.innerHTML = banner + data.results.map(r => `
          <div class="fnd-fc-item">
            <div class="fnd-fc-rating">
              ${r.type === "news" ? "📰" : "🔍"} ${r.rating}
            </div>
            <div class="fnd-fc-claim">${r.claim}</div>
            <div class="fnd-fc-meta">
              ${r.claimant !== "Related News"
                ? `By: ${r.claimant} · ` : ""}
              Source: ${r.publisher}
            </div>
            ${r.url
              ? `<a class="fnd-fc-link" href="${r.url}"
                    target="_blank">Read →</a>`
              : ""}
          </div>
        `).join("");

        // Show scroll hint after results load
        const hint = document.getElementById("fnd-scroll-hint");
        if (hint) hint.style.opacity = "1";
      }
    }
  );
}