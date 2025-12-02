# scripts/fetch_news.py
import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

# load .env
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")
if not API_KEY:
    raise ValueError("NEWS_API_KEY not found in .env")

OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

def fetch_latest_news():
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=100&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"news_raw_{timestamp}.json"
    path = os.path.join(OUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data.get('articles', []))} articles to {path}")
    return path

if __name__ == "__main__":
    fetch_latest_news()
