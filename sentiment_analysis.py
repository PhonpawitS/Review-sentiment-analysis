import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

try:
    data = pd.read_csv('IMDB Dataset.csv')
except FileNotFoundError:
    print("Not found the dataset file. Please ensure 'IMDB Dataset.csv' is in the current directory.")
    exit()

data['sentiment_label'] = data['sentiment'].map({'positive': 1, 'negative': 0})

X = data['cleaned_review']
y = data['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) 

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("Model training completed.")

y_pred = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

print("\n--- Model Test ---")

new_reviews = [
    "I absolutely loved this movie! The plot was thrilling and the characters were so well developed.",
    "This was a terrible film. I wasted two hours of my life watching it.",
    "An average movie with some good moments but overall not very memorable.",
    "Fantastic cinematography and a gripping storyline. Highly recommend it!",
    "The acting was subpar and the script was full of clichés."
]

new_reviews_tfidf = vectorizer.transform(new_reviews)

predictions = model.predict(new_reviews_tfidf)

for text, pred in zip(new_reviews, predictions):
    label = 'Positive' if pred == 1 else 'Negative'
    print(f"'{text}' \n   => Predict Answer: {label}\n")