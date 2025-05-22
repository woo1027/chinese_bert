import pymssql
import pandas as pd



# 建立連線
conn = pymssql.connect(
    server='192.168.0.104',
    user='sa',
    password='wu469711',
    database='video games'
)

# 撰寫 SQL 查詢語句
query = """
SELECT 
    l.student_id,
    COUNT(DISTINCT l.login_time) AS login_count,
    AVG(CAST(h.submitted AS FLOAT)) * 100 AS homework_completion_rate,
    AVG(e.score) AS avg_exam_score
FROM login_log l
LEFT JOIN homework_submission h ON l.student_id = h.student_id
LEFT JOIN exam_scores e ON l.student_id = e.student_id
GROUP BY l.student_id;
"""

df = pd.read_sql(query, conn)
conn.close()

from openpyxl import Workbook
from openpyxl.styles import PatternFill

wb = Workbook()
ws = wb.active
ws.title = "Student Report"

# 寫入標題列
ws.append(["Student ID", "Login Count", "HW Completion (%)", "Avg Exam Score"])

# 寫入數據
for row in df.itertuples(index=False):
    ws.append(list(row))

# 加入條件格式（例如低於60分標紅）
for row in ws.iter_rows(min_row=2, min_col=4, max_col=4):
    for cell in row:
        if cell.value is not None and cell.value < 60:
            cell.fill = PatternFill(start_color="FFC7CE", fill_type="solid")

wb.save("student_weekly_report.xlsx")

