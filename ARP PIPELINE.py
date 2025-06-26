import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def train_arp_detection_model_custom_features_fixed_split(labeled_csv_file, test_size=0.3, random_state=42):
    """
    Huấn luyện mô hình phát hiện tấn công ARP với các features tùy chỉnh
    và phân chia tập huấn luyện/kiểm tra cố định.
    """
    try:
        df = pd.read_csv("arpdata_labeled_legacy.csv")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{labeled_csv_file}'")
        return None, None, None
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return None, None, None

    if 'Label' not in df.columns:
        print("Lỗi: Cột 'Label' không tồn tại trong file CSV.")
        return None, None, None

    # Lọc chỉ giữ lại các gói tin ARP
    df_arp = df[df['Protocol'] == 'ARP'].copy()
    if df_arp.empty:
        print("Không có gói tin ARP nào để huấn luyện.")
        return None, None, None

    # Tạo feature ARP Type
    df_arp['ARP Type'] = df_arp['Info'].apply(lambda x: 1 if 'is at' in str(x) else 0 if 'Who has' in str(x) else -1)
    df_arp = df_arp[df_arp['ARP Type'] != -1].copy()

    # Tạo feature Is Broadcast Destination MAC
    df_arp['Is Broadcast Dest MAC'] = df_arp['Destination MAC'].apply(lambda x: 1 if str(x).lower() == 'ff:ff:ff:ff:ff:ff' else 0)

    # Xác định các features và cột mục tiêu
    features = ['Length', 'Source MAC Address', 'Destination MAC', 'ARP Type', 'Is Broadcast Dest MAC']
    categorical_features = ['Source MAC Address', 'Destination MAC']
    numerical_features = ['Length', 'ARP Type', 'Is Broadcast Dest MAC']
    target = 'Label'  # Sử dụng cột 'Label' đã được tạo trước đó

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra với kích thước cố định
    X_train, X_test, y_train, y_test = train_test_split(df_arp[features], df_arp[target], test_size=test_size, random_state=random_state, stratify=df_arp[target])

    print(f"Kích thước tập huấn luyện: {len(X_train)}")
    print(f"Kích thước tập kiểm tra: {len(X_test)}")

    # Tạo pipeline tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Fit preprocessor trên dữ liệu huấn luyện
    preprocessor.fit(X_train)

    # Tạo và huấn luyện mô hình
    model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))])

    model.fit(X_train, y_train)

    # Dự đoán và đánh giá
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Độ chính xác của mô hình trên tập kiểm tra: {accuracy:.4f}")
    print("\nBáo cáo phân loại:\n", classification_rep)
    print("\nMa trận nhầm lẫn:\n", conf_matrix)

    return model, preprocessor # Trả về cả mô hình và preprocessor

if __name__ == "__main__":
    labeled_file = 'arpdata_labeled.csv'  # Đảm bảo file này đã được tạo bởi code gán nhãn
    test_size_ratio = 0.3
    random_state_value = 42

    trained_model, preprocessor = train_arp_detection_model_custom_features_fixed_split(labeled_file, test_size_ratio, random_state_value)

    if trained_model and preprocessor:
        model_filename = 'modelarp_custom_features_fixed_split.joblib'
        joblib.dump(trained_model, model_filename)
        print(f"\nMô hình đã được lưu thành '{model_filename}'")

        preprocessor_filename = 'modelloi.joblib'
        joblib.dump(preprocessor, preprocessor_filename)
        print(f"Bộ tiền xử lý đã được lưu thành '{preprocessor_filename}'")