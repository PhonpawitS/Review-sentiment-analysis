import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

file_path = "mydatasets\imdb-dataset-of-50k-movie-reviews\IMDB Dataset.csv"
column = "review"

reviews = pd.read_csv(file_path, nrows=1000)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

reviews['cleaned_' + column] = reviews[column].apply(clean_text)

relative_path = file_path.split("mydatasets\\")[-1]

cleaned_df = reviews[['cleaned_' + column , "sentiment"]]
output_path = "mycleandatasets\\" + relative_path
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cleaned_df.to_csv(output_path, index=False)

print(f"Saved cleaned reviews to {output_path}")