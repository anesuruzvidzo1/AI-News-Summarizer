import streamlit as st
import pandas as pd
from pymongo import MongoClient
from joblib import load

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["news_db"]

processed_collection = db["processed_articles"]
classified_collection = db["classified_articles"]

# Load model + vectorizer
model = load("models/news_classifier_model.pkl")
vectorizer = load("models/tfidf_vectorizer.pkl")

# Classification function
def classify_text(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

# Re-run classification for all processed articles
def run_classification():
    articles = processed_collection.find()
    for article in articles:
        text = article.get("clean_text", "")
        if not text.strip():
            continue

        predicted = classify_text(text)

        classified_collection.replace_one(
            {"_id": article["_id"]},
            {
                "_id": article["_id"],
                "title": article.get("title", ""),
                "clean_text": text,
                "predicted_category": predicted,
                "publishedAt": article.get("publishedAt")
            },
            upsert=True
        )

# ---------------------- FRONT END ------------------------

st.title("📰 AI News Classifier & Summarizer")

# Run pipeline button
if st.button("Classify Latest Articles"):
    run_classification()
    st.success("Articles classified successfully!")

# Show category counts
st.header("📊 Classified Articles by Category")
categories = list(classified_collection.find({}, {"predicted_category": 1}))

if len(categories) > 0:
    df = pd.DataFrame(categories)
    counts = df["predicted_category"].value_counts()

    st.bar_chart(counts)



# Select category
st.header("📚 Browse Articles by Category")
selected_category = st.selectbox(
    "Choose a category:", 
    sorted(classified_collection.distinct("predicted_category"))
)

# Show summaries
st.subheader(f"Articles under: {selected_category}")

articles = classified_collection.find(
    {"predicted_category": selected_category},
    {"title": 1, "clean_text": 1}
)

for a in articles:
    st.write(f"### {a['title']}")
    st.write(a["clean_text"][:500] + "...")  
    st.write("---")
