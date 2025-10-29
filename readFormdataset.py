import pandas as pd

# สมมุติว่าไฟล์ชื่อ imdb_top_250.csv
file_path = "mydatasets\imdb-dataset-of-50k-movie-reviews\IMDB Dataset.csv"

# อ่านไฟล์ทั้งหมดเข้าเป็น DataFrame
df = pd.read_csv(file_path , usecols=["review"])

# แสดงข้อมูลบางส่วน (เช่น 5 แถวแรก)
print(df.iloc[1000])

a = str(df.iloc[1000])

print(a)