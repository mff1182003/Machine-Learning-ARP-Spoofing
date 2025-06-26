import pandas as pd
import random
from datetime import datetime, timedelta

def modify_arp_data(input_csv_file, output_csv_file, num_modified_packets=10, spoof_mac="00:aa:bb:cc:dd:ee", spoof_ip_prefix="192.168.1"):
    """
    Đọc file CSV ARP và tạo một file mới với một số gói tin ARP đã được thay đổi thông số.

    Args:
        input_csv_file (str): Đường dẫn đến file CSV ARP gốc.
        output_csv_file (str): Đường dẫn đến file CSV đã được thay đổi.
        num_modified_packets (int): Số lượng gói tin ARP muốn thay đổi.
        spoof_mac (str): Địa chỉ MAC giả mạo để thay thế.
        spoof_ip_prefix (str): Tiền tố IP giả mạo để thay thế (phần cuối sẽ được tạo ngẫu nhiên).
    """
    try:
        df = pd.read_csv('arpdata.csv')
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{input_csv_file}'")
        return

    arp_packets = df[df['Protocol'] == 'ARP'].copy()

    if arp_packets.empty:
        print("Không tìm thấy gói tin ARP nào trong file đầu vào.")
        return

    modified_packets = []
    indices_to_modify = random.sample(arp_packets.index.tolist(), min(num_modified_packets, len(arp_packets)))

    modified_df = df.copy()
    fake_date = datetime.now().strftime("%Y-%m-%d")  # Sử dụng ngày hiện tại làm giả định

    for index in indices_to_modify:
        modified_packet = modified_df.loc[index].copy()

        # Thay đổi MAC nguồn
        modified_packet['Source MAC Address'] = spoof_mac

        # Thay đổi IP nguồn (giữ nguyên tiền tố)
        modified_packet['Source IP ARP'] = f"{spoof_ip_prefix}.{random.randint(1, 254)}"

        # Thay đổi MAC đích (thường là MAC broadcast trong ARP request)
        modified_packet['Destination MAC'] = "ff:ff:ff:ff:ff:ff"

        # Thay đổi IP đích (thường là IP mục tiêu trong ARP request)
        modified_packet['Destination IP ARP'] = f"{spoof_ip_prefix}.{random.randint(100, 200)}" # Ví dụ: IP mục tiêu trong dải 100-200

        # Thay đổi thời gian một chút (tùy chọn)
        time_str = str(modified_packet['Time'])
        timestamp_format = "%H:%M:%S.%f" if '.' in time_str else "%H:%M:%S"
        try:
            current_time = datetime.strptime(time_str, timestamp_format)
            time_delta = timedelta(seconds=random.uniform(-2, 2))
            modified_packet['Time'] = f"{fake_date} {(current_time + time_delta).strftime('%H:%M:%S.%f')[:-3]}"
        except ValueError:
            try:
                current_time = datetime.strptime(time_str, "%H:%M:%S")
                time_delta = timedelta(seconds=random.uniform(-2, 2))
                modified_packet['Time'] = f"{fake_date} {(current_time + time_delta).strftime('%H:%M:%S.%f')[:-3]}"
            except ValueError:
                pass # Bỏ qua nếu không thể parse thời gian

        modified_df.loc[index] = modified_packet

    modified_df.to_csv(output_csv_file, index=False)
    print(f"Đã tạo file '{output_csv_file}' với {len(indices_to_modify)} gói tin ARP đã được thay đổi.")

if __name__ == "__main__":
    input_file = 'arpdata.csv'  # Thay thế bằng file ARP cũ của bạn
    output_file = 'arp_modified_data.csv'
    num_modify = 15  # Số lượng gói tin ARP muốn thay đổi
    fake_mac = "00:11:22:33:44:55"
    fake_ip_prefix = "192.168.10"

    modify_arp_data(input_file, output_file, num_modified_packets=num_modify, spoof_mac=fake_mac, spoof_ip_prefix=fake_ip_prefix)
    print(f"Bây giờ bạn có thể sử dụng file '{output_file}' để kiểm thử mô hình.")