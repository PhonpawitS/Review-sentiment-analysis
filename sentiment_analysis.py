import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# สมมติว่าไฟล์ CSV ของคุณมี 2 คอลัมน์: 'review_text' และ 'sentiment'
# 1. โหลดข้อมูล
try:
    data = pd.read_csv('IMDB Dataset.csv')
except FileNotFoundError:
    print("Not found the dataset file. Please ensure 'IMDB Dataset.csv' is in the current directory.")
    exit()

# 2. แปลง Label (ถ้ายังไม่ได้ทำ)
# สมมติ 'sentiment' เป็น "Positive" และ "Negative"
data['sentiment_label'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# 3. แบ่งข้อมูล Train/Test (แบ่งข้อมูล Text และ Label)
X = data['cleaned_review']
y = data['sentiment_label']

# แบ่งข้อมูล 80% เป็น Train, 20% เป็น Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. แปลงข้อความเป็นตัวเลข (Vectorization)
# เราจะใช้ TF-IDF
# เพิ่ม ngram_range=(1, 2) เพื่อให้มองทั้งคำเดี่ยว (1) และคำคู่ (2)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) # จำกัดที่ 5000 คำที่พบบ่อยสุด

# "เรียนรู้" คำศัพท์จากข้อมูล Train
X_train_tfidf = vectorizer.fit_transform(X_train)

# "แปลง" ข้อมูล Test โดยใช้คำศัพท์ที่เรียนรู้จาก Train
X_test_tfidf = vectorizer.transform(X_test)

# 5. เลือกและ Train โมเดล (ใช้ Naive Bayes)
# model = MultinomialNB()
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


print("Model training completed.")

# 6. ประเมินผล
y_pred = model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# --- ต่อจากโค้ดเดิม ---

# 7. ทดสอบโมเดลกับข้อความใหม่
print("\n--- Model Test ---")

# สร้างรีวิวตัวอย่าง
new_reviews = [
    "Fuck you movie.", # (Positive)
    "Movie is very good but i don't like it.",  # (Negative)
    "I want to wacth my granma instead of this movie", # (อาจจะทายยาก)
    "I love titanic." # (Positive)
]

# 1. ต้องแปลงข้อความใหม่ด้วย vectorizer *ตัวเดิม* ที่เรา Train ไว้
new_reviews_tfidf = vectorizer.transform(new_reviews)

# 2. สั่งให้โมเดลทำนาย
predictions = model.predict(new_reviews_tfidf)

# 3. แสดงผลลัพธ์ (1 คือ 'positive', 0 คือ 'negative' ที่เราตั้งไว้)
for text, pred in zip(new_reviews, predictions):
    label = 'Positive' if pred == 1 else 'Negative'
    print(f"'{text}' \n   => Predict Answer: {label}\n")