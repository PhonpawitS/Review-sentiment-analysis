import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    # ลบ <br /> หรือ <br> ออกจากข้อความ
    text = re.sub(r'<br\s*/?>', ' ', text)
    # ลบตัวอักษรที่ไม่ใช่ a-z หรือช่องว่าง
    text = re.sub(r'[^a-z\s]', '', text)
    # แยกเป็นคำ
    words = text.split()
    # ลบ stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    # ทำ stemming
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)



sample = "This movie!!! was sooo good :) 100% recommended."
print(clean_text(sample))