import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

file_path = "mydatasets\imdb-dataset-of-50k-movie-reviews\IMDB Dataset.csv"
column = "review"

reviews = pd.read_csv(file_path , usecols=[column], nrows=10)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

reviews['cleaned_review'] = reviews[column].apply(clean_text)

column_target = 1 

print(reviews.loc[column_target , "review"])
print("\n"+reviews.loc[column_target , "cleaned_review"])
