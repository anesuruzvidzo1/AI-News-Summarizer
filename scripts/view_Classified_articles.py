from pymongo import MongoClient
import pprint

client = MongoClient("mongodb://localhost:27017/")
db = client["news_db"]
classified_collection = db["classified_articles"]

# Fetch 30 classified articles
articles = classified_collection.find().limit(30)

for idx, article in enumerate(articles, 1):
    print(f"Article {idx}:")
    print(f"Title: {article.get('title')}")
    print(f"Category: {article.get('predicted_category')}")
    print(f"Published At: {article.get('publishedAt')}")
    print(f"Keywords: {article.get('keywords')}")
    print("-" * 60)
