import csv
import joblib
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import os
import glob

sender_email = "mff1182003@gmail.com"
sender_password = "kvcb dtvo umwr tdlr"
receiver_email = "danghuutoan1182003@gmail.com"
email_subject = "CẢNH BÁO TẤN CÔNG ARP SPOOFING NGHIÊM TRỌNG"
attack_threshold_count = 50  # 50 gói ARP Request liên tục


# THÊM CÁC HÀM TẠO FEATURES GIỐNG Y HỆT TRAINING CODE
def create_advanced_features(df):
    """
    HÀM Y HỆT TRONG TRAINING CODE - PHẢI GIỐNG 100%
    """
    df = df.copy()
    available_features = []

    # Features thời gian - chỉ nếu có Delta time
    if 'Delta time' in df.columns:
        # Log transform cho delta time để giảm skewness
        df['delta_time_log'] = np.log1p(df['Delta time'].clip(lower=0))
        available_features.append('delta_time_log')
        print(f"   ✅ delta_time_log: min={df['delta_time_log'].min():.4f}, max={df['delta_time_log'].max():.4f}")

    # Features về packet length
    if 'Length' in df.columns:
        length_mean = df['Length'].mean()
        length_std = df['Length'].std()
        if length_std > 0:
            df['length_zscore'] = (df['Length'] - length_mean) / length_std
            available_features.append('length_zscore')
            print(f"   ✅ length_zscore: min={df['length_zscore'].min():.4f}, max={df['length_zscore'].max():.4f}")

        # Binary indicators cho packet size
        q25 = df['Length'].quantile(0.25)
        q75 = df['Length'].quantile(0.75)
        df['is_small_packet'] = (df['Length'] <= q25).astype(int)
        df['is_large_packet'] = (df['Length'] >= q75).astype(int)
        available_features.extend(['is_small_packet', 'is_large_packet'])
        print(f"   ✅ packet_size_indicators: small={df['is_small_packet'].sum()}, large={df['is_large_packet'].sum()}")

    # Interaction features - chỉ tạo những cái đơn giản
    if 'Delta time' in df.columns and 'arp_type_numeric' in df.columns:
        df['arp_type_x_delta'] = df['arp_type_numeric'] * df['Delta time']
        available_features.append('arp_type_x_delta')
        print(f"   ✅ arp_type_x_delta: created")

    print(f"🔧 Đã tạo {len(available_features)} advanced features: {available_features}")
    return df


def preprocess_test_data(df):
    """
    Xử lý dữ liệu test GIỐNG HỆT training data
    """
    print("🔧 Bắt đầu xử lý dữ liệu test...")

    # Clean column names - loại bỏ khoảng trắng thừa
    df.columns = df.columns.str.strip()

    # Mapping tên cột từ CSV thực tế (nếu cần)
    column_mapping = {
        'No.': 'No',
        'Delta time': 'Delta time',
        'Frame length': 'Frame length'
    }

    # Rename nếu cần
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)

    # Tạo Delta time nếu không có
    if 'Delta time' not in df.columns:
        print("⚠️  Không có cột 'Delta time', tạo từ index")
        df['Delta time'] = df.index * 0.001  # Giả sử 1ms interval

    # Loại bỏ cột không cần thiết (chỉ những cột thực sự có)
    columns_to_drop = ['No', 'Time', 'Protocol', 'Tcp Flags', 'Time to live', 'Flags']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Tạo đặc trưng cơ bản CHỈ từ những gì có sẵn
    print("🔧 Tạo features cơ bản...")

    # ARP type từ Info
    df['arp_type_numeric'] = df['Info'].apply(lambda x: 1 if 'is at' in str(x) else 0 if 'Who has' in str(x) else -1)
    df = df[df['arp_type_numeric'] != -1].copy()
    print(f"   ✅ arp_type_numeric: {df['arp_type_numeric'].value_counts().to_dict()}")

    # Broadcast detection từ Info
    df['is_broadcast'] = df['Info'].apply(lambda x: 1 if 'Broadcast' in str(x) else 0)
    print(f"   ✅ is_broadcast: {df['is_broadcast'].value_counts().to_dict()}")

    # Source-Destination analysis nếu có
    if 'Source' in df.columns and 'Destination' in df.columns:
        df['same_src_dst'] = (df['Source'] == df['Destination']).astype(int)
        print(f"   ✅ same_src_dst: {df['same_src_dst'].value_counts().to_dict()}")
    else:
        print("   ⚠️  Không có cột Source/Destination")
        df['same_src_dst'] = 0

    # Tạo features nâng cao
    df = create_advanced_features(df)

    return df


def detect_consecutive_arp_requests(df, threshold=50):
    """
    Phát hiện 50 gói ARP Request "Who has" liên tục từ cùng một MAC address

    Args:
        df: DataFrame chứa dữ liệu ARP
        threshold: Số lượng ARP Request liên tục cần để cảnh báo

    Returns:
        DataFrame với cột Attack_Flag được đánh dấu
    """
    print(f"🔍 Phát hiện {threshold} ARP Request liên tục...")

    df = df.copy()
    df['Attack_Flag'] = 0

    # Chỉ xét các gói ARP Request (arp_type_numeric == 0)
    arp_requests = df[df['arp_type_numeric'] == 0].copy()
    print(f"📊 Tổng số ARP Request: {len(arp_requests)}")

    if len(arp_requests) == 0:
        print("⚠️  Không có ARP Request nào để kiểm tra")
        return df

    # Sắp xếp theo MAC address và thời gian
    sort_cols = ['Source MAC Address']
    if 'Time' in arp_requests.columns:
        sort_cols.append('Time')
    elif 'No.' in arp_requests.columns:
        sort_cols.append('No.')

    arp_requests = arp_requests.sort_values(by=sort_cols)

    # Phát hiện chuỗi liên tục cho từng MAC address
    attack_indices = []

    for mac_addr in arp_requests['Source MAC Address'].unique():
        mac_requests = arp_requests[arp_requests['Source MAC Address'] == mac_addr].copy()

        if len(mac_requests) < threshold:
            continue

        print(f"🔍 Kiểm tra MAC {mac_addr}: {len(mac_requests)} ARP Requests")

        # Tìm chuỗi liên tục dài nhất
        consecutive_count = 1
        max_consecutive = 1
        attack_start_idx = None

        for i in range(1, len(mac_requests)):
            current_idx = mac_requests.iloc[i].name
            prev_idx = mac_requests.iloc[i - 1].name

            # Kiểm tra xem có liên tục không (dựa trên index gần nhau)
            if current_idx - prev_idx <= 5:  # Cho phép tối đa 5 gói khác xen giữa
                consecutive_count += 1
                if consecutive_count >= threshold and attack_start_idx is None:
                    attack_start_idx = i - threshold + 1
            else:
                # Reset nếu bị ngắt quãng
                consecutive_count = 1
                attack_start_idx = None

            max_consecutive = max(max_consecutive, consecutive_count)

        print(f"   📈 Max consecutive requests: {max_consecutive}")

        # Đánh dấu tấn công nếu đạt ngưỡng
        if max_consecutive >= threshold and attack_start_idx is not None:
            # Đánh dấu tất cả các gói từ điểm bắt đầu tấn công
            attack_indices.extend(mac_requests.iloc[attack_start_idx:].index.tolist())
            print(f"   🚨 PHÁT HIỆN TẤNG CÔNG! {max_consecutive} ARP Requests liên tục")

    # Cập nhật Attack_Flag trong DataFrame gốc
    if attack_indices:
        df.loc[attack_indices, 'Attack_Flag'] = 1
        print(f"🚨 Tổng cộng {len(attack_indices)} gói được đánh dấu là tấn công")
    else:
        print("✅ Không phát hiện chuỗi ARP Request nghi ngờ")

    return df


# Load mô hình
try:
    loaded_model = joblib.load(r"C:\Users\Admin\PycharmProjects\ARP IDS\arp_spoofing_model_robust.joblib")
    print("✅ Đã load model thành công")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file mô hình.")
    exit()


# Hàm lấy file CSV mới nhất
def get_latest_csv_file(folder_path):
    list_of_files = glob.glob(os.path.join(folder_path, "capture_*.csv"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


csv_folder_path = 'C:/Users/Admin/Desktop/DataTest'
latest_csv_file = get_latest_csv_file(csv_folder_path)

if not latest_csv_file:
    print(f"Không tìm thấy file CSV nào trong thư mục '{csv_folder_path}'")
    exit()

print(f"Đang đọc file: {latest_csv_file}")
try:
    new_data = pd.read_csv(latest_csv_file, on_bad_lines='skip')
    new_data.columns = [col.strip() for col in new_data.columns]
    new_data.rename(columns={
        "frame.number": "No.",
        "frame.time": "Time",
        "ip.src": "Source",
        "ip.dst": "Destination",
        "_ws.col.Protocol": "Protocol",
        "frame.len": "Length",
        "_ws.col.Info": "Info",
        "tcp.flags.str": "Tcp Flags",
        "eth.src": "Source MAC Address",
        "eth.dst": "Destination MAC",
        "arp.src.proto_ipv4": "Source IP ARP",
        "arp.dst.proto_ipv4": "Destination IP ARP",
        "ip.ttl": "Time to live",
        "ip.flags": "Flags",
        "frame.time_delta": "Delta time"
    }, inplace=True)
    new_data = new_data[new_data['Protocol'] != 'MDNS']
    new_data = new_data[new_data['Destination'] != '224.0.0.251']
    print(f"✅ Đã đọc {len(new_data)} dòng dữ liệu")
except Exception as e:
    print(f"❌ Lỗi khi đọc/chuẩn hóa file CSV: {e}")
    exit()

# --- Xử lý dữ liệu ARP và dự đoán ---
new_data_arp = new_data[new_data['Protocol'] == 'ARP'].copy()
print(f"🔍 Tìm thấy {len(new_data_arp)} gói ARP")

if not new_data_arp.empty:
    # BƯỚC QUAN TRỌNG: Xử lý data giống training
    print("🔧 Xử lý dữ liệu ARP...")
    new_data_arp = preprocess_test_data(new_data_arp)

    # Đảm bảo có đủ tất cả features mà model cần
    required_features = [
        'Length', 'arp_type_numeric', 'Delta time',
        'is_broadcast', 'same_src_dst', 'delta_time_log',
        'length_zscore', 'is_small_packet', 'is_large_packet',
        'arp_type_x_delta'
    ]

    # Kiểm tra và thêm features thiếu
    missing_features = []
    for feature in required_features:
        if feature not in new_data_arp.columns:
            new_data_arp[feature] = 0  # Giá trị mặc định
            missing_features.append(feature)

    if missing_features:
        print(f"⚠️  Đã thêm {len(missing_features)} features thiếu: {missing_features}")

    # Chọn features theo đúng thứ tự mà model đã học
    X_predict = new_data_arp[required_features]

    print(f"📊 Features để dự đoán: {X_predict.shape}")
    print(f"📋 Danh sách features: {list(X_predict.columns)}")

    # --- Xử lý NaN ---
    if X_predict.isnull().values.any():
        print("⚠️ Dữ liệu chứa NaN, đang xử lý...")
        X_predict = X_predict.fillna(0)

    try:
        print("🔮 Bắt đầu dự đoán...")
        predictions = loaded_model.predict(X_predict)
        new_data_arp['Predict'] = predictions.astype(int)
        print(f"✅ Dự đoán thành công! Tìm thấy {sum(predictions)} gói nghi ngờ")

    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")
        print(f"Shape của X_predict: {X_predict.shape}")
        print(f"Columns của X_predict: {list(X_predict.columns)}")
        exit()

    # *** THAY ĐỔI CHÍNH: Sử dụng phương pháp phát hiện mới ***
    print(f"\n🔍 Phát hiện {attack_threshold_count} ARP Request liên tục...")
    new_data_arp = detect_consecutive_arp_requests(new_data_arp, attack_threshold_count)

    # Gửi email nếu có tấn công
    if new_data_arp['Attack_Flag'].any():
        attack_sources = new_data_arp.loc[new_data_arp['Attack_Flag'] == 1, 'Source MAC Address'].unique()

        print("🚨 Các địa chỉ MAC thực hiện ARP Request tấn công:")
        for mac in attack_sources:
            attack_count = len(new_data_arp[(new_data_arp['Source MAC Address'] == mac) &
                                            (new_data_arp['arp_type_numeric'] == 0)])
            print(f" - {mac}: {attack_count} ARP Requests")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"🚨 CẢNH BÁO ARP SPOOFING! 🚨\n\n"
        body += f"Thời gian phát hiện: {now}\n"
        body += f"Phát hiện {attack_threshold_count}+ ARP Request liên tục từ:\n\n"

        for mac in attack_sources:
            attack_count = len(new_data_arp[(new_data_arp['Source MAC Address'] == mac) &
                                            (new_data_arp['arp_type_numeric'] == 0)])
            body += f"📍 MAC: {mac}\n   Số ARP Requests: {attack_count}\n\n"

        body += "Khuyến nghị: Kiểm tra ngay thiết bị này để xác định có phải tấn công ARP Spoofing."

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                msg = MIMEText(body)
                msg['Subject'] = email_subject
                msg['From'] = sender_email
                msg['To'] = receiver_email
                server.sendmail(sender_email, receiver_email, msg.as_string())
            print("✅ Đã gửi email cảnh báo chi tiết.")
        except Exception as e:
            print(f"❌ Lỗi gửi email: {e}")
    else:
        print("✅ Không phát hiện chuỗi ARP Request nghi ngờ.")

    # Lưu kết quả gắn cờ
    try:
        merged = pd.merge(
            new_data,
            new_data_arp[['No.', 'Predict', 'Attack_Flag']],
            on='No.', how='left'
        )
        merged['Predict'] = merged['Predict'].fillna(0).astype(int)
        merged['Attack_Flag'] = merged['Attack_Flag'].fillna(0).astype(int)

        # Thống kê kết quả
        total_arp = len(new_data_arp)
        arp_requests = len(new_data_arp[new_data_arp['arp_type_numeric'] == 0])
        arp_replies = len(new_data_arp[new_data_arp['arp_type_numeric'] == 1])
        flagged_attacks = len(new_data_arp[new_data_arp['Attack_Flag'] == 1])

        print(f"\n📊 THỐNG KÊ PHÂN TÍCH:")
        print(f"   🔹 Tổng gói ARP: {total_arp}")
        print(f"   🔹 ARP Requests: {arp_requests}")
        print(f"   🔹 ARP Replies: {arp_replies}")
        print(f"   🔹 Gói bị đánh dấu tấn công: {flagged_attacks}")

        merged.to_csv('arpdata_predicted_with_consecutive_attack_detection.csv', index=False)
        print("✅ Đã lưu kết quả vào 'arpdata_predicted_with_consecutive_attack_detection.csv'")

    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")

else:
    print("⚠️  Không có gói ARP để phân tích.")