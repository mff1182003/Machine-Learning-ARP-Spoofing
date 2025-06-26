import pandas as pd
df = pd.read_csv('network_traffic_dataset_balanced.csv')
print("Phân bố nhãn trong toàn bộ DataFrame:")
print(df['Label'].value_counts())