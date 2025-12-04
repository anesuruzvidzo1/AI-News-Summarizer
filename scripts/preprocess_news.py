
import re
from pymongo import MongoClient
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


# Connection to our MongoDB with raw news articles for preprocessing

client = MongoClient("mongodb://localhost:27017/")
db = client["news_db"]

raw_collection = db["news_raw"]
processed_collection = db["processed_articles"]

# SpaCy NLP model
nlp = spacy.load("en_core_web_sm")



# Cleaning Functions for removal of url,html tags etc.

def clean_text(text):
    if text is None:
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Lowercase text
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text



#  NLP Processing 

def process_article(article):
   
    text = article.get("content") or article.get("description") or ""

    cleaned = clean_text(text)
    doc = nlp(cleaned)

    # Extract tokens (non-stopwords)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]

    # Extract lemmas
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    # POS Tags
    pos_tags = [{"token": token.text, "pos": token.pos_} for token in doc]

    # Named Entities
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)

    # Simple keyword extraction → top nouns
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN"]][:10]

    processed_record = {
        "_id": article["_id"],                # keep same ID as raw record
        "title": article.get("title", ""),
        "clean_text": cleaned,
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "entities": entities,
        "keywords": keywords,
        "publishedAt": article.get("publishedAt"),
        "source": article.get("source")
    }

    return processed_record



# Main Processing Pipeline

def run_preprocessing():

    # Get all raw articles not processed yet
    raw_articles = raw_collection.find()

    count = 0
    for article in raw_articles:
        # Skip if already processed
        if processed_collection.find_one({"_id": article["_id"]}):
            continue

        processed_record = process_article(article)
        processed_collection.insert_one(processed_record)

        count += 1
        if count % 50 == 0:
            print(f"Processed {count} articles...")

    print(f"Total articles processed: {count}")


# Run script
if __name__ == "__main__":
    run_preprocessing()
