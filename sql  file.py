import pymssql
import csv
import pandas as pd
import os
import numpy as np

host = '192.168.0.104'
database = 'video games'
username = 'sa'
password = 'wu469711'

conn = pymssql.connect(
    server = host,
    user = username,
    password = password,
    database = database
)

# cursor = conn.cursor()
# cursor.execute('Select name From sys.databases')
# for row in cursor.fetchall():
#     print(row)
#
# cursor.close()
# conn.close()
cursor = conn.cursor()



# CSV 檔案路徑
df = pd.read_csv('D:/面試資料/R/vgsales_utf8.csv')


# 顯示缺失值數量
print(df.isna().sum())
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Global_Sales'] = pd.to_numeric(df['Global_Sales'], errors='coerce')
df.dropna(subset=['Year', 'Global_Sales','Publisher'], inplace=True)

cursor.execute("""
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='game' AND xtype='U')
CREATE TABLE game (
    Rank INT IDENTITY(1,1) PRIMARY KEY,
    Name NVARCHAR(255),
    Platform NVARCHAR(50),
    Year INT,
    Genre NVARCHAR(50),
    Publisher NVARCHAR(255),
    NA_Sales FLOAT,
    EU_Sales FLOAT,
    JP_Sales FLOAT,
    Other_Sales FLOAT,
    Global_Sales FLOAT
)
""")
conn.commit()


# 準備插入資料的 SQL 語法（使用參數化）
insert_sql = """
INSERT INTO game (Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
# 將 DataFrame 轉成要插入的列表（跳過 Rank，因為是自動遞增）
data_to_insert = df[['Name', 'Platform', 'Year',
                     'Genre', 'Publisher', 'NA_Sales', 'EU_Sales',
                     'JP_Sales', 'Other_Sales', 'Global_Sales']].values.tolist()

# 批量插入
cursor.executemany(insert_sql, data_to_insert)
conn.commit()

print("資料已成功匯入 MSSQL 資料庫！")

cursor.close()
conn.close()

