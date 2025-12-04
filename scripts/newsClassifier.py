
import os
import sys
import logging
from pymongo import MongoClient
from joblib import load
from sklearn.exceptions import NotFittedError

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
db = client.get_database(os.getenv("MONGO_DB", "news_db"))

processed_collection = db["processed_articles"]
classified_collection = db["classified_articles"]

# Load model + vectorizer 
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../models/news_classifier_model.pkl")
vectorizer_path = os.path.join(script_dir, "../models/tfidf_vectorizer.pkl")

try:
    model = load(model_path)
    vectorizer = load(vectorizer_path)
    logging.info(f"Loaded model from {model_path} and vectorizer from {vectorizer_path}")
except FileNotFoundError as e:
    logging.error(f"Model or vectorizer not found: {e}")
    sys.exit(1)
except Exception as e:
    logging.exception("Failed to load model/vectorizer:")
    sys.exit(1)

# Classification function 
def classify_articles():
    articles = processed_collection.find()
    count_total = 0
    count_classified = 0
    count_skipped = 0
    count_errors = 0

    for article in articles:
        count_total += 1
        try:
            # Prefer cleaned text; fallback to content/description/title
            text = article.get("clean_text") or article.get("content") or article.get("description") or article.get("title") or ""
            if not str(text).strip():
                count_skipped += 1
                logging.debug(f"Skipping _id={article.get('_id')} (empty text)")
                continue

            # Transform and predict
            X = vectorizer.transform([text])
            predicted_category = model.predict(X)[0]

            # Build classified document including url and summaries
            classified_doc = {
                "_id": article["_id"],
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "clean_text": text,
                "summary": article.get("summary", ""),
                "summary_short": article.get("summary_short", ""),
                "predicted_category": predicted_category,
                "keywords": article.get("keywords", []),
                "entities": article.get("entities", {}),
                "publishedAt": article.get("publishedAt")
            }

            # Upsert into classified collection
            classified_collection.replace_one({"_id": article["_id"]}, classified_doc, upsert=True)
            count_classified += 1

        except NotFittedError as e:
            logging.error("Model/vectorizer not fitted correctly: %s", e)
            count_errors += 1
        except Exception as e:
            logging.exception("Error classifying _id=%s: %s", article.get("_id"), e)
            count_errors += 1

    logging.info("Summary: processed=%d classified=%d skipped=%d errors=%d",
                 count_total, count_classified, count_skipped, count_errors)
    print(f"classified {count_classified} articles! ({count_skipped} skipped, {count_errors} errors)")

#  Run if executed directly 
if __name__ == "__main__":
    classify_articles()
