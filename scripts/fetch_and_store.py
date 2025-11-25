import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient, errors
import hashlib

# load env
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "news_db")
COLLECTION = os.getenv("MONGO_COLLECTION", "news_raw")

if not API_KEY:
    raise RuntimeError("NEWS_API_KEY not found in .env")

# helper: make unique id from article URL or content
def article_id(article):
    url = article.get("url") or article.get("title") or json.dumps(article, sort_keys=True)
    return hashlib.sha1(url.encode("utf-8")).hexdigest()

def fetch_articles():
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=100&apiKey={API_KEY}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("articles", [])

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION]
    # create unique index on _id to prevent duplicates (id will be _id)
    try:
        col.create_index("url", unique=True, sparse=True)
    except Exception:
        pass

    articles = fetch_articles()
    inserted = 0
    skipped = 0
    for a in articles:
        # add metadata
        a["_fetched_at"] = datetime.utcnow().isoformat()
        # choose unique key: prefer url, else use shard from content
        uid = a.get("url")
        if not uid:
            uid = article_id(a)
            a["_synthetic_id"] = uid

        # try insert and catch duplicate
        try:
            # if URL exists, insert with that unique field; Mongo will enforce unique index
            col.insert_one({**a})
            inserted += 1
        except errors.DuplicateKeyError:
            skipped += 1
        except Exception as e:
            # fallback: try upsert using synthetic id
            if "_synthetic_id" in a:
                try:
                    col.update_one({"_synthetic_id": a["_synthetic_id"]}, {"$setOnInsert": a}, upsert=True)
                    inserted += 1
                except errors.DuplicateKeyError:
                    skipped += 1
            else:
                print("Insert error:", e)

    print(f"done. inserted={inserted}, skipped={skipped}, total_fetched={len(articles)}")
    client.close()

if __name__ == "__main__":
    main()
