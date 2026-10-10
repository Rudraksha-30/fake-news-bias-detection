import requests
import os
from dotenv import load_dotenv

load_dotenv()

FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
NEWS_API_KEY       = os.getenv("NEWS_API_KEY")
FACT_CHECK_URL     = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
NEWS_API_URL       = "https://newsapi.org/v2/everything"

def extract_query(text: str) -> str:
    """Extract key terms from text for better API matching."""
    stop_words = {"is", "the", "a", "an", "and", "or", "but",
                  "in", "on", "at", "to", "for", "of", "with",
                  "that", "this", "it", "was", "are", "were",
                  "has", "have", "had", "be", "been", "being",
                  "i", "we", "they", "he", "she", "as", "by"}
    words = text.split()
    keywords = [w.strip(".,!?\"'") for w in words
                if w.lower().strip(".,!?\"'") not in stop_words]
    return " ".join(keywords[:6])

def search_fact_check(query: str, max_results: int = 3):
    """
    Search Google Fact Check API first.
    If empty, fall back to NewsAPI for related articles.
    """
    # Step 1: Try Google Fact Check
    fact_results = _google_fact_check(query, max_results)
    if fact_results.get("results"):
        return fact_results

    # Step 2: Fall back to NewsAPI
    news_results = _news_api_search(query, max_results)
    if news_results.get("results"):
        return news_results

    # Step 3: Nothing found
    return {
        "results": [],
        "source":  "none",
        "message": "No fact-checks or related news found."
    }

def _google_fact_check(query: str, max_results: int):
    """Query Google Fact Check Tools API."""
    if not FACT_CHECK_API_KEY:
        return {"results": [], "source": "factcheck"}

    search_query = extract_query(query)
    print(f"Google Fact Check query: '{search_query}'")

    try:
        params = {
            "query":        search_query,
            "key":          FACT_CHECK_API_KEY,
            "pageSize":     max_results,
            "languageCode": "en"
        }
        response = requests.get(FACT_CHECK_URL, params=params, timeout=5)
        data     = response.json()
        print(f"Google Fact Check status: {response.status_code}")

        if "claims" not in data or not data["claims"]:
            print("Google Fact Check: no claims found")
            return {"results": [], "source": "factcheck"}

        results = []
        for claim in data["claims"][:max_results]:
            review = claim.get("claimReview", [{}])[0]
            results.append({
                "type":      "factcheck",
                "claim":     claim.get("text", ""),
                "claimant":  claim.get("claimant", "Unknown"),
                "rating":    review.get("textualRating", "Unknown"),
                "publisher": review.get("publisher", {}).get("name", "Unknown"),
                "url":       review.get("url", ""),
                "date":      review.get("reviewDate", "")[:10]
                             if review.get("reviewDate") else ""
            })

        print(f"Google Fact Check: found {len(results)} results")
        return {"results": results, "source": "factcheck"}

    except Exception as e:
        print(f"Google Fact Check error: {e}")
        return {"results": [], "source": "factcheck"}

def _news_api_search(query: str, max_results: int = 3):
    """Search NewsAPI for related articles as fallback."""
    if not NEWS_API_KEY:
        print("NewsAPI key not configured")
        return {"results": [], "source": "news"}

    search_query = extract_query(query)
    print(f"NewsAPI query: '{search_query}'")

    try:
        params = {
            "q":        search_query,
            "apiKey":   NEWS_API_KEY,
            "pageSize": max_results,
            "language": "en",
            "sortBy":   "relevancy"
        }
        response = requests.get(NEWS_API_URL, params=params, timeout=5)
        data     = response.json()
        print(f"NewsAPI status: {response.status_code}")
        print(f"NewsAPI total results: {data.get('totalResults', 0)}")

        articles = data.get("articles", [])
        if not articles:
            print("NewsAPI: no articles found")
            return {"results": [], "source": "news"}

        results = []
        for article in articles[:max_results]:
            # Skip removed articles
            if article.get("title") == "[Removed]":
                continue
            results.append({
                "type":      "news",
                "claim":     article.get("title", "")[:200],
                "claimant":  article.get("author") or "Unknown",
                "rating":    "Related News Article",
                "publisher": article.get("source", {}).get("name", "Unknown"),
                "url":       article.get("url", ""),
                "date":      article.get("publishedAt", "")[:10]
                             if article.get("publishedAt") else ""
            })

        print(f"NewsAPI: found {len(results)} articles")
        return {
            "results": results,
            "source":  "news",
            "message": "No direct fact-checks found. "
                       "Showing related news articles instead."
        }

    except Exception as e:
        print(f"NewsAPI error: {e}")
        return {"results": [], "source": "news"}