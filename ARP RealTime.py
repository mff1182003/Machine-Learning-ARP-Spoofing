import joblib
import pandas as pd

# Tải mô hình đã lưu
loaded_model = joblib.load('modelarp_custom_features_fixed_split.joblib')

# Đọc file CSV dữ liệu mới bạn muốn dự đoán
new_data = pd.read_csv('arpdata.csv')

# **Thực hiện các bước tiền xử lý tương tự như trên dữ liệu huấn luyện:**
# Lọc gói tin ARP
new_data_arp = new_data[new_data['Protocol'] == 'ARP'].copy()
if not new_data_arp.empty:
    # Tạo feature ARP Type
    new_data_arp['ARP Type'] = new_data_arp['Info'].apply(lambda x: 1 if 'is at' in str(x) else 0 if 'Who has' in str(x) else -1)
    new_data_arp = new_data_arp[new_data_arp['ARP Type'] != -1].copy()

    # Tạo feature Is Broadcast Destination MAC
    new_data_arp['Is Broadcast Dest MAC'] = new_data_arp['Destination MAC'].apply(lambda x: 1 if str(x).lower() == 'ff:ff:ff:ff:ff:ff' else 0)

    # Chọn các features đã sử dụng để huấn luyện
    predict_features = ['Length', 'Source MAC Address', 'Destination MAC', 'ARP Type', 'Is Broadcast Dest MAC']
    X_predict = new_data_arp[predict_features]

    # Dự đoán nhãn cho dữ liệu mới
    predictions = loaded_model.predict(X_predict)

    # In ra các dự đoán
    print("Dự đoán nhãn cho dữ liệu mới:")
    print(predictions)

    # Bạn có thể kết hợp các dự đoán này với DataFrame ban đầu nếu cần
    predictions_df = pd.DataFrame({'Predicted_Label': predictions})
    results_df = pd.concat([new_data_arp.reset_index(drop=True), predictions_df], axis=1)
    print("\nDataFrame kết hợp với nhãn dự đoán:")
    print(results_df)

else:
    print("Không có gói tin ARP nào trong file dữ liệu để dự đoán.")