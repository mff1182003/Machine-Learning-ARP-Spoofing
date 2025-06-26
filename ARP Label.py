import pandas as pd

def label_arp_spoofing_legacy(csv_file):
    try:
        df = pd.read_csv("arpdata.csv")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{csv_file}'")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return None

    df['Label'] = 0

    # --- Feature 1: Số lượng lớn ARP Request từ một nguồn đến các đích khác nhau ---
    arp_who_has_df = df[df['Info'].str.contains('Who has', na=False)].copy()
    if not arp_who_has_df.empty:
        source_request_counts = arp_who_has_df.groupby('Source MAC Address')['Destination IP ARP'].nunique()
        suspicious_sources_many_requests = source_request_counts[source_request_counts > 50].index.tolist()
        df.loc[df['Source MAC Address'].isin(suspicious_sources_many_requests) & df['Info'].str.contains('Who has', na=False), 'Label'] = 1

    # --- Feature 2: Gratuitous ARP đáng ngờ (dựa trên IP nguồn = IP đích) ---
    gratuitous_arp_df = df[
        (df['Protocol'] == 'ARP') &
        (df['Info'].str.contains('is at', na=False)) &
        (df['Source IP ARP'] == df['Destination IP ARP'])
    ].index
    df.loc[gratuitous_arp_df, 'Label'] = 1

    # --- Feature 3: Thay đổi MAC Address cho một IP (dựa trên quan sát trong toàn bộ log) ---
    arp_reply_df = df[(df['Protocol'] == 'ARP') & (df['Info'].str.contains('is at', na=False))].copy()
    if not arp_reply_df.empty:
        ip_mac_mapping = arp_reply_df.groupby('Source IP ARP')['Source MAC Address'].nunique()
        suspicious_ips_mac_change = ip_mac_mapping[ip_mac_mapping > 2].index.tolist()
        df.loc[df['Source IP ARP'].isin(suspicious_ips_mac_change) & df['Info'].str.contains('is at', na=False), 'Label'] = 1

    print("Đã gán nhãn tấn công (dựa trên các features có sẵn) vào cột 'Label'.")
    print(df['Label'].value_counts())
    print(df.head())

    return df

if __name__ == "__main__":
    file_path = 'arpdata.csv'
    labeled_df = label_arp_spoofing_legacy(file_path)

    if labeled_df is not None:
        labeled_df.to_csv('arpdata_labeled_legacy.csv', index=False)
        print("Đã lưu file đã gán nhãn thành 'arpdata_labeled_legacy.csv'")