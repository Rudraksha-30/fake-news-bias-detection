// Create right-click context menu on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id:       "checkFakeNews",
    title:    "🔍 Check with Fake News Detector",
    contexts: ["selection"]   // only shows when text is selected
  });
});

// When user clicks the context menu
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

  // Send loading state to content script
  chrome.tabs.sendMessage(tab.id, { type: "SHOW_LOADING" });

  try {
    const response = await fetch("http://localhost:5000/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text: selectedText })
    });

    const data = await response.json();
    chrome.tabs.sendMessage(tab.id, { type: "SHOW_RESULT", data });

  } catch (err) {
    chrome.tabs.sendMessage(tab.id, {
      type:  "SHOW_RESULT",
      error: "Could not connect to API. Make sure Flask server is running."
    });
  }
});