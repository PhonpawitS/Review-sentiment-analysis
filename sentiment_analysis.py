import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# สมมติว่าไฟล์ CSV ของคุณมี 2 คอลัมน์: 'review_text' และ 'sentiment'
# 1. โหลดข้อมูล
try:
    data = pd.read_csv('IMDB Dataset.csv')
except FileNotFoundError:
    print("Not found the dataset file. Please ensure 'IMDB Dataset.csv' is in the current directory.")
    exit()

# 2. แปลง Label (ถ้ายังไม่ได้ทำ)
# สมมติ 'sentiment' เป็น "Positive" และ "Negative"
data['sentiment_label'] = data['sentiment'].map({'Positive': 1, 'Negative': 0})

# 3. แบ่งข้อมูล Train/Test (แบ่งข้อมูล Text และ Label)
X = data['review_text']
y = data['sentiment_label']

# แบ่งข้อมูล 80% เป็น Train, 20% เป็น Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. แปลงข้อความเป็นตัวเลข (Vectorization)
# เราจะใช้ TF-IDF
vectorizer = TfidfVectorizer(max_features=5000) # จำกัดที่ 5000 คำที่พบบ่อยสุด

# "เรียนรู้" คำศัพท์จากข้อมูล Train
X_train_tfidf = vectorizer.fit_transform(X_train)

# "แปลง" ข้อมูล Test โดยใช้คำศัพท์ที่เรียนรู้จาก Train
X_test_tfidf = vectorizer.transform(X_test)

# 5. เลือกและ Train โมเดล (ใช้ Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("โมเดล Train เสร็จสิ้น!")

# 6. ประเมินผล
y_pred = model.predict(X_test_tfidf)

print(f"ความแม่นยำ (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))