# Machine Learning ARP Spoofing Detection

This project is an ARP Spoofing detection system using Machine Learning. It collects real-time network traffic, processes it into structured data, labels it, and detects spoofing attacks using a trained model.

---

## 📌 Prerequisites

### ✅ 1. Install Python & PyCharm

- Download Python 3.x: https://www.python.org/downloads/
- Download PyCharm: https://www.jetbrains.com/pycharm/download/

---

### ✅ 2. Required Python Libraries

Install the following libraries:

```bash
pip install pandas numpy scikit-learn joblib
📡 Packet Capturing & Feature Extraction
✅ 3. Install Wireshark
Download Wireshark: https://www.wireshark.org/download.html

⚙️ Configure Wireshark Columns:
Go to Preferences → Appearance → Columns, and add/enable the following columns (as in the provided example file):

No.

Time

Source

Destination

Protocol

Length

Info

ARP opcode

HW type

Sender MAC address

Sender IP address

Target MAC address

Target IP address

These are important for identifying ARP patterns.

📥 Export Packets to CSV:
Start capturing packets on your active network interface.

Stop after capturing enough ARP traffic.

Go to File → Export Packet Dissections → As CSV.

Save as your_capture.csv.

This file can be used directly with the system. No need to process it with CICFlowMeter.

🏷️ Labeling Data
Place your exported .csv file into the Label/ directory.

Run the labeling script:

bash
Sao chép
Chỉnh sửa
python Label.py
This will clean, label, and standardize the data format for detection.

🛡️ Detection System
After labeling is done, run the detection:

bash
Sao chép
Chỉnh sửa
python CheckStatus.py
This will:

Load your labeled CSV file.

Use the pre-trained Random Forest model to classify each row.

If spoofing is detected, an email alert will be sent.

✉️ Email Alert Configuration
Open the file CheckStatus.py, and fill in these variables:

python
Sao chép
Chỉnh sửa
sender_email = "your_sender_email@gmail.com"
receiver_email = "your_receiver_email@gmail.com"
app_password = "your_gmail_app_password"
You must generate a Gmail App Password (not your regular password) here:
https://myaccount.google.com/apppasswords
(Requires 2-Step Verification enabled)

✅ Done!
You're now ready to detect ARP spoofing attacks using machine learning and real traffic data!

📂 Folder Structure (Example)
css
Sao chép
Chỉnh sửa
Machine-Learning-ARP-Spoofing/
├── Label.py
├── CheckStatus.py
├── model.pkl
├── Label/
│   └── captured_data.csv
🧠 Model
The model used is a Random Forest Classifier trained on labeled ARP spoofing data.

Retraining script can be added separately.
