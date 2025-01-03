import pandas as pd
import sqlite3
from net_teach import train_and_predict_lstm

conn = sqlite3.connect('instance/data.db')

# Извлечение данных с помощью pandas
df = pd.read_sql_query("SELECT * FROM data", conn)
print(df)
# Закрытие соединения с базой данных
conn.close()

output = train_and_predict_lstm(df)
print(output)