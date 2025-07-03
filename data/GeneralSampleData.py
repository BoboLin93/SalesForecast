import pandas as pd
import numpy as np

# 模擬參數
np.random.seed(123)
clients = ['C001', 'C002', 'C003']
opportunities = ['O001', 'O002', 'O003']
dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq='D')

# 建立假資料
data = []
for client in clients:
    for opportunity in opportunities:
        order_dates = np.random.choice(dates, size=60, replace=False)
        order_dates = sorted(order_dates)
        for date in order_dates:
            order_amount = np.random.poisson(lam=100) + np.random.randint(-15, 15)
            data.append({
                "datetime": pd.to_datetime(date).strftime("%Y-%m-%d"),
                "client": client,
                "opportunity": opportunity,
                "order_amount": max(order_amount, 0)
            })

# 轉換為 DataFrame 並輸出為 CSV
df = pd.DataFrame(data)
df = df.sort_values(by=["client", "opportunity", "datetime"])
df.reset_index(drop=True, inplace=True)
df.to_csv("sample_sales_data.csv", index=False)

print(df.head())