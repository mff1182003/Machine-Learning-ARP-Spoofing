import joblib
import pandas as pd

# Tải mô hình đã lưu
loaded_model = joblib.load('arp_spoofing_model_cleaned.joblib')

# Đọc file CSV dữ liệu mới bạn muốn dự đoán
input_file = 'arp.csv'  # Thay thế bằng tên file đầu vào của bạn
output_file = 'testarppredicted2.csv'  # Tên file CSV đầu ra

try:
    new_data = pd.read_csv('testarppredicted.csv')
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{input_file}'")
    exit()

# **Thực hiện các bước tiền xử lý tương tự như trên dữ liệu huấn luyện:**
# Lọc gói tin ARP
new_data_arp = new_data[new_data['Protocol'] == 'ARP'].copy()

if not new_data_arp.empty:
    # Tạo feature arp_type_numeric
    new_data_arp['arp_type_numeric'] = new_data_arp['Info'].apply(lambda x: 1 if 'is at' in str(x) else 0 if 'Who has' in str(x) else -1)
    new_data_arp = new_data_arp[new_data_arp['arp_type_numeric'] != -1].copy()

    # Tạo feature mac_mismatch
    new_data_arp['mac_mismatch'] = (new_data_arp['Source'] != new_data_arp['Source MAC Address']).astype(int)

    # Chọn các features đã sử dụng để huấn luyện (phải khớp với code huấn luyện)
    predict_features = ['Length', 'Source', 'Destination', 'Source MAC Address', 'Destination MAC', 'Source IP ARP', 'Destination IP ARP', 'Delta time', 'arp_type_numeric', 'mac_mismatch']

    # Kiểm tra xem các cột features có tồn tại trong DataFrame hay không
    missing_cols = [col for col in predict_features if col not in new_data_arp.columns]
    if missing_cols:
        print(f"Lỗi: Các cột sau không tồn tại trong file dự đoán: {missing_cols}")
        print(f"Các cột hiện có trong dữ liệu ARP: {new_data_arp.columns.tolist()}")
    else:
        X_predict = new_data_arp[predict_features]

        # Dự đoán nhãn cho dữ liệu mới
        predictions = loaded_model.predict(X_predict)

        # Tạo cột 'Predict' dựa trên kết quả dự đoán
        new_data_arp['Predict'] = predictions.astype(int)

        # Kết hợp cột 'Predict' với DataFrame ban đầu (nếu cần giữ lại các gói tin không phải ARP)
        merged_df = pd.merge(new_data, new_data_arp[['No.', 'Predict']], on='No.', how='left')
        merged_df['Predict'] = merged_df['Predict'].fillna(0).astype(int) # Gán 0 cho các gói không phải ARP

        # Lưu DataFrame kết quả vào một file CSV mới
        merged_df.to_csv(output_file, index=False)
        print(f"Đã lưu kết quả dự đoán vào file '{output_file}'")

else:
    # Nếu không có gói tin ARP nào trong file đầu vào
    new_data['Predict'] = 0
    new_data.to_csv(output_file, index=False)
    print(f"Không có gói tin ARP nào để dự đoán. Đã lưu file '{output_file}' với cột 'Predict' toàn là 0.")