import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load and Sample Dataset
CSV_PATH = "/Users/russell/Downloads/raw-data.csv"   

df = pd.read_csv(CSV_PATH)

# Only keep the columns we need
df = df[["content", "description", "category"]]

#Drop rows where the category is missing
df = df.dropna(subset=["category"])

# Categories and topics we want
target_categories = [
    "Art", "Artificial Intelligence", "Climate", "Cryptocurrency", "Education",
    "Fashion", "Fitness", "Food", "Health", "Music", "Politics",
    "Science", "Stock", "Technology", "Weather", "Sports", "Movies", "Finance", 
    "Bitcoin"
]

# Filter dataset to keep only these categories
df = df[df["category"].isin(target_categories)]

# Combine description + content 
df["text"] = df["content"].fillna("") + " " + df["description"].fillna("")

# Remove empty text rows
df["text"] = df["text"].str.strip()
df = df[df["text"] !=""]

# SAMPLE the dataset 
df = df.groupby("category").sample(n=2000, random_state=42)

print(f"Dataset size after sampling: {len(df)}")

#Cleaning Function 

def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Keep only letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text



df["clean_text"] = df["text"].apply(clean_text)



# Lemmatization 


nlp = spacy.load("en_core_web_sm")

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

df["lemmatized"] = df["clean_text"].apply(lemmatize)



# TF-IDF Vectorizer

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(df["lemmatized"])
y = df["category"]



# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Train Logistic Regression Classifier
model = LogisticRegression(max_iter=300, n_jobs=-1)
model.fit(X_train, y_train)

from joblib import dump

dump(model, "news_classifier_model.pkl")
dump(vectorizer, "tfidf_vectorizer.pkl")


# Evaluate Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}\n")
print(classification_report(y_test, y_pred))

