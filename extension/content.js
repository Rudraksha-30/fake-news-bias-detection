let popup = null;

// Listen for messages from background.js
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "SHOW_LOADING") {
    showPopup(null, null, true);
  }
  if (msg.type === "SHOW_RESULT") {
    if (msg.error) {
      showPopup(null, msg.error, false);
    } else {
      showPopup(msg.data, null, false);
    }
  }
});

function showPopup(data, error, loading) {
  // Remove existing popup
  if (popup) popup.remove();

  popup = document.createElement("div");
  popup.id = "fnd-popup";

  if (loading) {
    popup.innerHTML = `
      <div class="fnd-header">
        <span>🔍 Fake News Detector</span>
        <button class="fnd-close" onclick="this.closest('#fnd-popup').remove()">✕</button>
      </div>
      <div class="fnd-body">
        <div class="fnd-loading">Analyzing text...</div>
      </div>
    `;
  } else if (error) {
    popup.innerHTML = `
      <div class="fnd-header">
        <span>🔍 Fake News Detector</span>
        <button class="fnd-close" onclick="this.closest('#fnd-popup').remove()">✕</button>
      </div>
      <div class="fnd-body">
        <div class="fnd-error">${error}</div>
      </div>
    `;
  } else {
    const icons = { FAKE: "🔴", REAL: "🟢", UNCERTAIN: "🟡" };
    const colors = { FAKE: "#ef4444", REAL: "#22c55e", UNCERTAIN: "#f59e0b" };

    popup.innerHTML = `
      <div class="fnd-header">
        <span>🔍 Fake News Detector</span>
        <button class="fnd-close" onclick="this.closest('#fnd-popup').remove()">✕</button>
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
              <div class="fnd-bar-fill" style="width:${data.real_prob}%;background:#22c55e"></div>
            </div>
            <span>${data.real_prob}%</span>
          </div>
          <div class="fnd-bar-row">
            <span>🔴 FAKE</span>
            <div class="fnd-bar-bg">
              <div class="fnd-bar-fill" style="width:${data.fake_prob}%;background:#ef4444"></div>
            </div>
            <span>${data.fake_prob}%</span>
          </div>
        </div>
      </div>
    `;
  }

  document.body.appendChild(popup);

  // Auto close after 10 seconds
  setTimeout(() => { if (popup) popup.remove(); }, 10000);
}