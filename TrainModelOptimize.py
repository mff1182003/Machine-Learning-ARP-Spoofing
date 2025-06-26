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
    """Thêm nhiễu Gaussian vào features số"""
    np.random.seed(random_state)
    X_noisy = X.copy()

    # Chỉ thêm nhiễu cho features số
    numerical_cols = ['Length', 'Delta time', 'arp_type_numeric']

    for col in numerical_cols:
        if col in X_noisy.columns:
            noise = np.random.normal(0, noise_factor * X_noisy[col].std(), size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise

    return X_noisy


def create_advanced_features(df):
    """Tạo features phức tạp hơn để phát hiện ARP spoofing"""
    df = df.copy()

    # Features thời gian
    df['delta_time_log'] = np.log1p(df['Delta time'])
    df['delta_time_squared'] = df['Delta time'] ** 2

    # Features về packet length
    df['length_zscore'] = (df['Length'] - df['Length'].mean()) / df['Length'].std()
    df['is_small_packet'] = (df['Length'] < df['Length'].quantile(0.25)).astype(int)
    df['is_large_packet'] = (df['Length'] > df['Length'].quantile(0.75)).astype(int)

    # Features về tần suất
    if 'Source' in df.columns:
        source_counts = df['Source'].value_counts()
        df['source_frequency'] = df['Source'].map(source_counts)
        df['is_frequent_source'] = (df['source_frequency'] > df['source_frequency'].median()).astype(int)

    # Interaction features
    df['arp_type_x_delta'] = df['arp_type_numeric'] * df['Delta time']
    df['length_x_delta'] = df['Length'] * df['Delta time']

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

    # Loại bỏ cột không cần thiết
    columns_to_drop = ['No.', 'Time', 'Protocol', 'Tcp Flags', 'Time to live', 'Flags', 'Frame length']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    # Tạo đặc trưng cơ bản
    df['arp_type_numeric'] = df['Info'].apply(lambda x: 1 if 'is at' in str(x) else 0 if 'Who has' in str(x) else -1)
    df = df[df['arp_type_numeric'] != -1].copy()

    if 'Source MAC Address' in df.columns and 'Source' in df.columns:
        df['mac_mismatch'] = (df['Source'] != df['Source MAC Address']).astype(int)
    else:
        df['mac_mismatch'] = 0

    # Tạo features nâng cao
    df = create_advanced_features(df)

    # Chọn features
    base_features = ['Length', 'Delta time', 'arp_type_numeric', 'mac_mismatch']
    advanced_features = ['delta_time_log', 'length_zscore', 'is_small_packet',
                         'is_large_packet', 'arp_type_x_delta']

    # Chỉ sử dụng features có trong DataFrame
    features = [f for f in base_features + advanced_features if f in df.columns]
    print(f"🔧 Sử dụng {len(features)} features: {features}")

    X = df[features]
    y = df['Label']

    # Thêm nhiễu để tăng tính robust
    print(f"🎲 Thêm nhiễu với hệ số {noise_factor}")
    X_noisy = add_noise_to_features(X, noise_factor=noise_factor, random_state=random_state)

    # Tách dữ liệu với stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_noisy, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"📦 Tập huấn luyện: {len(X_train)} | Tập kiểm tra: {len(X_test)}")

    # Tách features cho preprocessing
    numerical_features = [f for f in features if f not in ['mac_mismatch', 'is_small_packet',
                                                           'is_large_packet', 'is_frequent_source']]
    binary_features = [f for f in features if f in ['mac_mismatch', 'is_small_packet',
                                                    'is_large_packet', 'is_frequent_source']]

    # Pipeline với regularization mạnh hơn
    pipeline = Pipeline([
        ('preprocessing', ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('bin', 'passthrough', binary_features)
        ])),
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