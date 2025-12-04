
import os, re
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "news_db")
PROC_COLL = os.getenv("PROCESSED_COLLECTION", "processed_articles")

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text):
    if not text:
        return []
    text = re.sub(r'\s+', ' ', text).strip()
    sents = _SENTENCE_SPLIT_RE.split(text)
    sents = [s.strip() for s in sents if len(s.strip()) > 20]
    return sents

def summarize_text_tfidf(text, top_n_sentences=3):
    sents = split_sentences(text)
    if not sents:
        return "", ""
    if len(sents) <= top_n_sentences:
        summary = " ".join(sents)
        short = sents[0] if sents else ""
        return summary, short
    vect = TfidfVectorizer(stop_words='english')
    X = vect.fit_transform(sents)
    scores = X.sum(axis=1).A1
    # pick top n indices and preserve original order
    top_idx = np.argsort(scores)[-top_n_sentences:]
    top_idx = sorted(top_idx)
    summary = " ".join([sents[i] for i in top_idx])
    # short summary: top single sentence
    short_idx = int(np.argmax(scores))
    summary_short = sents[short_idx]
    return summary, summary_short

def main(batch_size=500, top_n=3):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    proc = db[PROC_COLL]

    # find processed docs missing summary
    query = {"$or": [{"summary": {"$exists": False}}, {"summary": ""}, {"summary": None},
                     {"summary_short": {"$exists": False}}, {"summary_short": None}, {"summary_short": ""}]}
    cursor = proc.find(query).limit(batch_size)

    count = 0
    for doc in cursor:
        text = doc.get("clean_text") or doc.get("content") or doc.get("description") or doc.get("title") or ""
        if not text.strip():
            continue
        summary, short = summarize_text_tfidf(text, top_n_sentences=top_n)
        if not summary.strip():
            continue
        proc.update_one({"_id": doc["_id"]}, {"$set": {"summary": summary, "summary_short": short}})
        count += 1
    print(f"Summarized {count} articles.")
    client.close()

if __name__ == "__main__":
    main()
