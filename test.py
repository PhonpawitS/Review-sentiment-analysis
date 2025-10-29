import kagglehub
import os
import pandas as pd
import shutil

# ดาวน์โหลด dataset
path = kagglehub.dataset_download("mohamedasak/imdb-top-250-movies")
print("Path ที่โหลดมา:", path)

# ย้ายไปยัง path ที่ต้องการ
target_path = "mydatasets/imdb_top250"
os.makedirs(target_path, exist_ok=True)
shutil.copytree(path, target_path, dirs_exist_ok=True)
print("ย้าย dataset ไปที่:", target_path)

# แสดงไฟล์ทั้งหมดในโฟลเดอร์ใหม่
files = os.listdir(target_path)
print("Files in dataset folder:", files)

# หาว่าไฟล์ไหนเป็น .csv แล้วอ่านออกมา
for file in files:
    if file.endswith(".csv"):
        csv_path = os.path.join(target_path, file)
        df = pd.read_csv(csv_path)
        print("\n📄 อ่านข้อมูลจาก:", csv_path)
        print(df.head())  # แสดง 5 แถวแรกของข้อมูล
        break
else:
    print("❌ ไม่พบไฟล์ CSV ในโฟลเดอร์นี้")
