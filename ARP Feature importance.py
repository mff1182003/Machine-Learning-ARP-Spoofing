import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import matplotlib.pyplot as plt

def train_and_show_feature_importance(csv_file):
    try:
        df = pd.read_csv("arpdata_labeled_legacy.csv")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{csv_file}'")
        return

    # Lọc chỉ giữ lại các gói tin ARP
    df_arp = df[df['Protocol'] == 'ARP'].copy()
    if df_arp.empty:
        print("Không có gói tin ARP nào để huấn luyện.")
        return

    # Feature Engineering
    df_arp['ARP Type'] = df_arp['Info'].apply(lambda x: 1 if 'is at' in str(x) else 0 if 'Who has' in str(x) else -1)
    df_arp = df_arp[df_arp['ARP Type'] != -1].copy()
    df_arp['Is Broadcast Dest MAC'] = df_arp['Destination MAC'].apply(lambda x: 1 if str(x).lower() == 'ff:ff:ff:ff:ff:ff' else 0)

    # Chọn features và target
    features = ['Length', 'Source MAC Address', 'Destination MAC', 'Source IP ARP', 'Destination IP ARP', 'ARP Type', 'Is Broadcast Dest MAC']
    categorical_features = ['Source MAC Address', 'Destination MAC', 'Source IP ARP', 'Destination IP ARP']
    numerical_features = ['Length', 'ARP Type', 'Is Broadcast Dest MAC']
    target = 'Label'

    # Loại bỏ các hàng có giá trị NaN trong các features quan trọng
    df_arp = df_arp[features + [target]].dropna()

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(df_arp[features], df_arp[target], test_size=0.2, random_state=42, stratify=df_arp[target])

    # Tạo pipeline tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Tạo và huấn luyện mô hình Random Forest
    model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

    model.fit(X_train, y_train)

    # Lấy độ quan trọng của features
    if hasattr(model['classifier'], 'feature_importances_'):
        importances = model['classifier'].feature_importances_
        feature_names = preprocessor.get_feature_names_out(features)

        # Sắp xếp độ quan trọng theo thứ tự giảm dần
        indices = np.argsort(importances)[::-1]

        # Hiển thị biểu đồ
        num_encoded_features = len(feature_names)
        num_features_to_plot = min(num_encoded_features, 15)  # Hiển thị tối đa 15 features

        plt.figure(figsize=(12, 6))
        plt.title("Độ quan trọng của các Features trong mô hình ARP Spoofing")
        plt.bar(range(num_features_to_plot), importances[indices[:num_features_to_plot]], align="center")
        plt.xticks(range(num_features_to_plot), feature_names[indices[:num_features_to_plot]], rotation=90, ha='right')
        plt.xlabel("Feature")
        plt.ylabel("Độ quan trọng")
        plt.tight_layout()
        plt.show()
    else:
        print("Mô hình không hỗ trợ thuộc tính 'feature_importances_'.")

if __name__ == "__main__":
    wireshark_file = 'arpdata.csv'  # **SỬA ĐƯỜNG DẪN TỚI FILE ĐÃ GÁN NHÃN CỦA BẠN**
    train_and_show_feature_importance(wireshark_file)