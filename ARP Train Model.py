import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def add_noise_to_features(X, noise_factor=0.1, random_state=42):
    """Thêm nhiễu Gaussian vào features số - CHỈ cho features thực sự numerical"""
    np.random.seed(random_state)
    X_noisy = X.copy()

    # Danh sách features thực sự là số
    numerical_indicators = ['Length', 'Delta time', 'delta_time', 'length_', '_x_', 'zscore']

    noise_applied = []
    for col in X_noisy.columns:
        # Kiểm tra xem có phải numerical feature không
        is_numerical = any(indicator in col for indicator in numerical_indicators)
        # Kiểm tra variance > 0
        has_variance = X_noisy[col].std() > 1e-6

        if is_numerical and has_variance:
            noise = np.random.normal(0, noise_factor * X_noisy[col].std(), size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise
            noise_applied.append(col)

    print(f"🎲 Đã thêm nhiễu cho {len(noise_applied)} features: {noise_applied}")
    return X_noisy


def create_advanced_features(df):
    """Tạo features CHỈ từ dữ liệu có sẵn - không tạo features phức tạp"""
    df = df.copy()

    # Chỉ tạo features đơn giản từ dữ liệu có sẵn
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


def plot_learning_curves(pipeline, X, y, cv_folds=5):
    """Vẽ learning curves để phát hiện overfitting"""
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=cv_folds, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1_macro'
    )

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
    plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.3)
    plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.3)

    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_scores, val_scores


def train_arp_detection_model_robust(labeled_csv_file, test_size=0.2, cv_folds=10,
                                     noise_factor=0.15, random_state=42):
    """
    Phiên bản cải tiến với kỹ thuật chống overfitting
    """
    try:
        df = pd.read_csv(labeled_csv_file)
        print(f"📊 Đã đọc {len(df)} mẫu dữ liệu")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file CSV: {e}")
        return None, None, None

    if 'Label' not in df.columns:
        print("❌ Lỗi: Thiếu cột 'Label'")
        print(f"Các cột có sẵn: {df.columns.tolist()}")
        return None, None, None

    # Phân tích phân bố class
    print(f"📈 Phân bố nhãn: {dict(df['Label'].value_counts())}")

    # In ra tên các cột để debug
    print(f"🔍 Các cột trong file: {df.columns.tolist()}")

    # Clean column names - loại bỏ khoảng trắng thừa
    df.columns = df.columns.str.strip()

    # Mapping tên cột từ CSV thực tế
    column_mapping = {
        'No.': 'No',
        'Delta time': 'Delta time',
        'Frame length': 'Frame length'
    }

    # Rename nếu cần
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)

    # Kiểm tra các cột cần thiết
    required_columns = ['Info', 'Length', 'Label']
    optional_columns = ['Delta time', 'Source', 'Destination']

    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        print(f"❌ Thiếu các cột bắt buộc: {missing_required}")
        return None, None, None

    available_optional = [col for col in optional_columns if col in df.columns]
    print(f"✅ Các cột tùy chọn có sẵn: {available_optional}")

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

    # Tạo features nâng cao
    df = create_advanced_features(df)

    # Chọn features CHỈ từ những gì thực sự có trong DataFrame
    base_features = ['Length', 'arp_type_numeric']
    if 'Delta time' in df.columns:
        base_features.append('Delta time')

    # Optional features
    optional_features = ['is_broadcast']
    if 'same_src_dst' in df.columns:
        optional_features.append('same_src_dst')

    # Advanced features (chỉ những cái được tạo thành công)
    possible_advanced = ['delta_time_log', 'length_zscore', 'is_small_packet',
                         'is_large_packet', 'arp_type_x_delta']

    # Tổng hợp tất cả features có thể
    all_possible_features = base_features + optional_features + possible_advanced

    # CHỈ sử dụng features thực sự có trong DataFrame
    features = [f for f in all_possible_features if f in df.columns]

    print(f"🔧 SỬ DỤNG {len(features)} FEATURES:")
    for i, feature in enumerate(features, 1):
        print(f"   {i:2d}. {feature}")

    # Kiểm tra data quality cho từng feature
    print(f"\n📊 THỐNG KÊ FEATURES:")
    for feature in features:
        if df[feature].dtype in ['int64', 'float64']:
            print(
                f"   {feature:20s}: min={df[feature].min():8.4f}, max={df[feature].max():8.4f}, mean={df[feature].mean():8.4f}")
        else:
            print(f"   {feature:20s}: unique_values={df[feature].nunique()}")

    X = df[features]
    y = df['Label']

    print(f"\n📦 FINAL DATASET: {X.shape[0]} samples x {X.shape[1]} features")

    # Thêm nhiễu để tăng tính robust
    print(f"🎲 Thêm nhiễu với hệ số {noise_factor}")
    X_noisy = add_noise_to_features(X, noise_factor=noise_factor, random_state=random_state)

    # Tách dữ liệu với stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"📦 Tập huấn luyện: {len(X_train)} | Tập kiểm tra: {len(X_test)}")

    # Tách features cho preprocessing - CHỈ dựa trên tên thực tế
    numerical_features = []
    binary_features = []

    for feature in features:
        # Numerical: Length, Delta time, log transforms, z-scores, interactions
        if any(keyword in feature.lower() for keyword in ['length', 'delta', 'log', 'zscore', '_x_']):
            numerical_features.append(feature)
        # Binary: is_, arp_type_numeric (0/1), same_
        elif any(keyword in feature.lower() for keyword in ['is_', 'arp_type_numeric', 'same_']):
            binary_features.append(feature)
        else:
            # Default to numerical for safety
            numerical_features.append(feature)

    print(f"📊 PREPROCESSING SETUP:")
    print(f"   Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"   Binary features ({len(binary_features)}): {binary_features}")

    # Tạo preprocessing pipeline
    if binary_features:
        preprocessing = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('bin', 'passthrough', binary_features)
        ])
    else:
        preprocessing = ColumnTransformer([
            ('num', StandardScaler(), numerical_features)
        ])
        print("⚠️  Chỉ có numerical features, không có binary features")

    # Pipeline với regularization mạnh hơn
    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('clf', RandomForestClassifier(
            n_estimators=100,  # Giảm số cây
            max_depth=5,  # Giới hạn độ sâu
            min_samples_split=20,  # Tăng min samples để split
            min_samples_leaf=10,  # Tăng min samples ở leaf
            max_features='sqrt',  # Giảm số features mỗi split
            random_state=random_state,
            class_weight='balanced'  # Cân bằng class
        ))
    ])

    # Cross-validation chi tiết với nhiều fold
    print(f"🔄 Thực hiện {cv_folds}-fold cross validation...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)

    print(f"📊 CV F1 Scores: {cv_scores}")
    print(f"📊 Mean CV F1: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

    # Validation curve để kiểm tra overfitting
    print("📈 Phân tích validation curve...")
    param_range = [3, 5, 7, 10, 15, 20]
    train_scores, val_scores = validation_curve(
        pipeline, X_train, y_train, param_name='clf__max_depth',
        param_range=param_range, cv=5, scoring='f1_macro', n_jobs=-1
    )

    # Tìm max_depth tối ưu
    val_mean = np.mean(val_scores, axis=1)
    optimal_depth = param_range[np.argmax(val_mean)]
    print(f"🏆 Optimal max_depth: {optimal_depth}")

    # Set optimal depth và train final model
    pipeline.set_params(clf__max_depth=optimal_depth)

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"✅ Huấn luyện hoàn tất sau {training_time:.2f} giây")

    # Đánh giá chi tiết
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    print(f"\n🎯 KẾT QUẢ ĐÁNH GIÁ:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Training F1: {train_f1:.4f}")
    print(f"   Test F1: {test_f1:.4f}")
    print(f"   Accuracy Gap: {train_acc - test_acc:.4f}")
    print(f"   F1 Gap: {train_f1 - test_f1:.4f}")

    if (train_acc - test_acc) > 0.1 or (train_f1 - test_f1) > 0.1:
        print("⚠️  CẢNH BÁO: Có dấu hiệu overfitting!")
    else:
        print("✅ Model có vẻ ổn định, không overfitting nghiêm trọng")

    print(f"\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    print("\n🧮 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    # Feature importance
    if hasattr(pipeline.named_steps['clf'], 'feature_importances_'):
        importances = pipeline.named_steps['clf'].feature_importances_
        feature_imp = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\n🔍 TOP 5 FEATURES QUAN TRỌNG:")
        print(feature_imp.head())

    # Lưu mô hình
    joblib.dump(pipeline, 'arp_spoofing_model_robust.joblib')
    print("✅ Đã lưu mô hình: arp_spoofing_model_robust.joblib")

    # Vẽ learning curves (nếu có matplotlib)
    try:
        plot_learning_curves(pipeline, X_train, y_train, cv_folds=5)
    except:
        print("📊 Không thể vẽ learning curves (thiếu matplotlib)")

    results = {
        'cv_scores': cv_scores,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'optimal_depth': optimal_depth,
        'feature_importance': feature_imp if 'feature_imp' in locals() else None
    }

    return pipeline, results


# Chạy với cấu hình chống overfitting
if __name__ == "__main__":
    print("🚀 Bắt đầu training với kỹ thuật chống overfitting...")

    model, results = train_arp_detection_model_robust(
        'arp.csv',
        test_size=0.2,  # Giảm test size để có nhiều data train hơn
        cv_folds=10,  # Tăng số fold
        noise_factor=0.15,  # Thêm nhiễu 15%
        random_state=42
    )

    if results:
        print(f"\n📊 TỔNG KẾT:")
        print(f"CV Mean F1: {results['cv_scores'].mean():.4f}")
        print(f"Test F1: {results['test_f1']:.4f}")
        print(
            f"Overfitting Check: {'❌ Có overfitting' if results['train_f1'] - results['test_f1'] > 0.1 else '✅ Ổn định'}")