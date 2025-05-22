import kagglehub
import pandas as pd
import os
from sqlalchemy import create_engine



# Download latest version
path = kagglehub.dataset_download("supergus/multistage-continuousflow-manufacturing-process")


# 取得該資料夾中的所有檔案
files = os.listdir(path)
print(files)

# 讀取一個 .csv 文件（假設其中一個叫 'process_data.csv'）
df = pd.read_csv(os.path.join(path, "continuous_factory_process.csv"))
print(df.head())

# 替換以下資訊
user = "sqlprostudio-ro"
password = "Yes"
host = "35.82.3.43"
port = 3306
database = "northwind"

# 建立連線引擎
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

# 寫入資料（如果 table 已存在可以使用 if_exists='replace'）
df.to_sql("process_data", con=engine, if_exists="replace", index=False)

print("✅ 寫入完成！")