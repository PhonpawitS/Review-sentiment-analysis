import kagglehub
import os
import pandas as pd

# ดาวน์โหลด dataset
path = kagglehub.dataset_download("mohamedasak/imdb-top-250-movies")
print("Path to dataset files:", path)

# แสดงไฟล์ทั้งหมดในโฟลเดอร์ที่ดาวน์โหลดมา
files = os.listdir(path)
print("Files in dataset folder:", files)

# สมมุติว่าไฟล์หลักชื่อ "imdb_top_250.csv" (หรือชื่อใกล้เคียง)
for file in files:
    if file.endswith(".csv"):
        csv_path = os.path.join(path, file)
        df = pd.read_csv(csv_path)
        print("\n📄 อ่านข้อมูลจาก:", csv_path)
        print(df.head())  # แสดง 5 แถวแรกของข้อมูล
        break
else:
    print("❌ ไม่พบไฟล์ CSV ในโฟลเดอร์นี้")
