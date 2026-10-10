// ── Context menu setup ────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id:       "checkFakeNews",
      title:    "🔍 Check with Fake News Detector",
      contexts: ["selection"]
    });
  });
});

// ── Context menu click handler ────────────────────────────
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "checkFakeNews") return;

  const selectedText = info.selectionText.trim();
  if (!selectedText || selectedText.length < 10) {
    chrome.tabs.sendMessage(tab.id, {
      type:  "SHOW_RESULT",
      error: "Please select at least 10 characters."
    });
    return;
  }

  chrome.tabs.sendMessage(tab.id, { type: "SHOW_LOADING" });

  try {
    const response = await fetch("http://localhost:5000/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text: selectedText })
    });

    const data = await response.json();
    data.input_text = selectedText;
    chrome.tabs.sendMessage(tab.id, { type: "SHOW_RESULT", data });

  } catch (err) {
    chrome.tabs.sendMessage(tab.id, {
      type:  "SHOW_RESULT",
      error: "Could not connect to API. Make sure Flask server is running."
    });
  }
});

// ── Handle API requests from content.js ──────────────────
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type !== "API_REQUEST") return;

  fetch(`http://localhost:5000${msg.endpoint}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(msg.body)
  })
  .then(res  => res.json())
  .then(data => sendResponse({ data }))
  .catch(err => sendResponse({ error: err.message }));

  return true; // keeps message channel open for async response
});