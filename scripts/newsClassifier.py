import spacy
from pymongo import MongoClient
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer


#  CONNECT TO MONGODB

client = MongoClient("mongodb://localhost:27017/")
db = client["news_db"]

processed_collection = db["processed_articles"]
classified_collection = db["classified_articles"]


# LOAD MODEL + TF-IDF VECTORIZER

model = load("../models/news_classifier_model.pkl")
vectorizer = load("../models/tfidf_vectorizer.pkl")


# 
# CLASSIFY CLEANED ARTICLES
# 
def classify_articles():
    
    
    articles = processed_collection.find()  
    
    count = 0
    for article in articles:

        text = article.get("clean_text", "")
        if not text.strip():
            continue

        # Convert clean text → TF-IDF vector
        X = vectorizer.transform([text])

        # Predict category
        predicted_category = model.predict(X)[0]

        # Prepare new document
        classified_doc = {
            "_id": article["_id"],
            "title": article.get("title", ""),
            "clean_text": text,
            "predicted_category": predicted_category,
            "keywords": article.get("keywords", []),
            "entities": article.get("entities", {}),
            "publishedAt": article.get("publishedAt")
        }

        # Insert into classified collection
        classified_collection.replace_one(
            {"_id": article["_id"]},   # update if exists
            classified_doc,
            upsert=True
        )

        count += 1
    
    print(f"classified {count} articles!")


# RUN SCRIPT
if __name__ == "__main__":
    classify_articles()
