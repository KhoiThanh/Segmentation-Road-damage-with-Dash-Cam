# Running Guide & Hướng dẫn chạy - RDD YOLO Segmentation

This repository contains guides in both English and Vietnamese.
*Tài liệu này bao gồm hướng dẫn bằng cả tiếng Anh và tiếng Việt.*

- [English Version](#english-version)
- [Bản tiếng Việt](#bản-tiếng-việt)

---

# English Version

This document provides instructions on how to install and run the components of the Road Damage Detection (RDD) project using YOLOv11-seg.

The project consists of 3 main components:
1. **Web App (Gradio)**: An intuitive web interface for testing image and video detection, and viewing Grad-CAM/Eigen-CAM heatmaps.
2. **Mobile App Backend (FastAPI)**: An API server providing detection endpoints and storing history data.
3. **Mobile App Client (Flutter)**: An Android mobile application that connects to the Backend.

---

## 1. Guide to Running the Web App (Gradio)

The Web App runs locally on your computer using the Gradio library.

### Installation Requirements:
Install the necessary Python libraries:
```bash
pip install gradio ultralytics opencv-python numpy torch
```

### How to Run:
1. Open a terminal and navigate to the project's root directory:
   ```powershell
   cd D:\Document\Code\DACN\RDDYOLOseg
   ```
2. Run the `web_app.py` file:
   ```powershell
   python web_app.py
   ```
3. Once running, the terminal will display a local URL (typically `http://127.0.0.1:7860`). Open a web browser and access this address.
4. The Web interface supports:
   * **Image Detection**: Upload images to draw masks/bounding boxes and display Eigen-CAM heatmaps.
   * **Video Detection**: Process video frames and output the resulting video with statistics.
   * **Eigen-CAM Heatmap**: Visualize the feature regions that the model focuses on.

---

## 2. Guide to Running the Mobile App Backend (FastAPI)

The Backend is responsible for processing real-time detection requests via WebSocket and HTTP sent from mobile devices.

### Installation Requirements:
1. Navigate to the backend directory:
   ```powershell
   cd D:\Document\Code\DACN\RDDYOLOseg\mobile_app\backend
   ```
2. Create a virtual environment and install libraries:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

### How to Run:
* **Option 1**: Quick run using the pre-configured `.bat` script:
  ```powershell
  .\run.bat
  ```
* **Option 2**: Run manually using the `uvicorn` command:
  ```powershell
  $env:RDD_MODEL_PATH = "D:\Document\Code\DACN\RDDYOLOseg\runs\segment\Road_Patches_Cracks_Segmentation_Datasets_yolo11l\weights\best.pt"
  python -m uvicorn main:app --host 0.0.0.0 --port 8000
  ```
* **Verify Operation**: Access `http://localhost:8000/health` in your computer's browser to check the API response status.

---

## 3. Connection Configuration for Mobile App

To allow the Flutter app on your phone to send images/videos to the backend on your computer, you need to configure the network connection in one of the following ways:

### Option 1: Using Local Wifi (Same LAN)
* On the computer running the backend, open a terminal and type `ipconfig` to get your **IPv4** address (e.g., `192.168.1.15`).
* The backend address on the mobile phone will be: `http://192.168.1.15:8000`.

### Option 2: Exposing to the Internet via Ngrok (Recommended for real-world road tests)
1. Download and configure Ngrok at [ngrok.com](https://ngrok.com/).
2. Open a new terminal and run:
   ```powershell
   ngrok http 8000
   ```
3. Copy the URL in the format `https://xxxx-xxx.ngrok-free.app` and paste it into the mobile app's settings.

---

## 4. Guide to Running the Mobile App Client (Flutter)

The Android application, written in Flutter, is installed and run on an Android device or emulator.

### Preparation Requirements:
* Install **Flutter SDK** and ensure `flutter doctor` runs successfully (especially the Android toolchain section).
* Enable **USB Debugging** on your Android phone and connect it to your computer with a USB cable.

### How to Run:
1. Open a terminal in the Flutter project directory:
   ```powershell
   cd D:\Document\Code\DACN\RDDYOLOseg\mobile_app\flutter_app
   ```
2. Initialize the Android platform directory (only required the first time):
   ```powershell
   flutter create . --project-name rdd_mobile --org com.dacn.rdd --platforms android
   ```
3. Download package dependencies and run the app:
   ```powershell
   flutter pub get
   flutter run
   ```
4. If you want to build a release APK file to install directly on the phone:
   ```powershell
   flutter build apk --release
   ```
   * The resulting APK file will be located at: `build/app/outputs/flutter-apk/app-release.apk`.

---

## 5. Using the Mobile Application
1. Launch the mobile app, tap the **Settings (⚙️)** icon in the top right corner of the screen.
2. Paste the backend URL (from **Section 3**) and tap **Save and test connection**.
3. Once the connection is successful, return to the main screen to test the features:
   * **Camera/Gallery**: Capture or select road images to send for detection, with support for saving history and GPS coordinates.
   * **Real-time Camera**: Stream live video from the camera via WebSocket to the server and display real-time segmentation results.
   * **Upload Video**: Upload a video and receive the processed segmentation video.
   * **History & Map**: View historical records of cracks and potholes along with GPS coordinates on Google Maps.

---

# Bản tiếng Việt

Tài liệu này hướng dẫn cách cài đặt và chạy các thành phần trong dự án Nhận diện Ổ gà và Vết nứt (Road Damage Detection - RDD) sử dụng YOLOv11-seg.

Dự án bao gồm 3 thành phần chính:
1. **Web App (Gradio)**: Giao diện Web trực quan dùng để thử nghiệm nhận diện ảnh, video và xem bản đồ nhiệt Grad-CAM/Eigen-CAM.
2. **Mobile App Backend (FastAPI)**: Máy chủ API cung cấp các endpoint nhận diện và lưu trữ dữ liệu lịch sử.
3. **Mobile App Client (Flutter)**: Ứng dụng Android chạy trên thiết bị di động kết nối với Backend.

---

## 1. Hướng dẫn chạy Web App (Gradio)

Web App chạy trực tiếp trên máy tính thông qua thư viện Gradio.

### Yêu cầu cài đặt:
Cài đặt các thư viện Python cần thiết:
```bash
pip install gradio ultralytics opencv-python numpy torch
```

### Cách chạy:
1. Mở terminal và di chuyển đến thư mục gốc của dự án:
   ```powershell
   cd D:\Document\Code\DACN\RDDYOLOseg
   ```
2. Khởi chạy file `web_app.py`:
   ```powershell
   python web_app.py
   ```
3. Sau khi chạy, terminal sẽ hiện lên đường dẫn cục bộ (thường là `http://127.0.0.1:7860`). Mở trình duyệt web và truy cập địa chỉ này.
4. Giao diện Web hỗ trợ:
   * **Nhận diện Hình ảnh**: Tải ảnh lên để vẽ mask/bounding box và hiển thị Eigen-CAM heatmap.
   * **Nhận diện Video**: Xử lý từng frame video và xuất ra video kết quả kèm thống kê.
   * **Eigen-CAM Heatmap**: Trực quan hóa vùng đặc trưng mà mô hình tập trung chú ý.

---

## 2. Hướng dẫn chạy Mobile App Backend (FastAPI)

Backend chịu trách nhiệm xử lý các yêu cầu nhận dạng thời gian thực qua WebSocket và HTTP gửi từ thiết bị di động.

### Yêu cầu cài đặt:
1. Di chuyển vào thư mục backend:
   ```powershell
   cd D:\Document\Code\DACN\RDDYOLOseg\mobile_app\backend
   ```
2. Tạo môi trường ảo và cài đặt thư viện:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

### Cách chạy:
* **Cách 1**: Chạy nhanh bằng file script `.bat` đã tạo sẵn:
  ```powershell
  .\run.bat
  ```
* **Cách 2**: Chạy thủ công qua lệnh `uvicorn`:
  ```powershell
  $env:RDD_MODEL_PATH = "D:\Document\Code\DACN\RDDYOLOseg\runs\segment\Road_Patches_Cracks_Segmentation_Datasets_yolo11l\weights\best.pt"
  python -m uvicorn main:app --host 0.0.0.0 --port 8000
  ```
* **Kiểm tra hoạt động**: Truy cập `http://localhost:8000/health` trên trình duyệt máy tính để xem trạng thái phản hồi của API.

---

## 3. Cấu hình kết nối cho Mobile App

Để ứng dụng Flutter trên điện thoại gửi được ảnh/video tới backend trên máy tính, bạn cần cấu hình liên kết mạng theo một trong hai cách:

### Cách 1: Sử dụng mạng Wifi nội bộ (Cùng mạng LAN)
* Trên máy tính chạy backend, mở terminal gõ `ipconfig` để lấy địa chỉ **IPv4** (ví dụ: `192.168.1.15`).
* Địa chỉ backend trên điện thoại sẽ là: `http://192.168.1.15:8000`.

### Cách 2: Chia sẻ ra ngoài Internet qua Ngrok (Khuyên dùng khi demo thực tế ngoài đường)
1. Tải và cấu hình Ngrok tại [ngrok.com](https://ngrok.com/).
2. Mở một terminal mới và chạy lệnh:
   ```powershell
   ngrok http 8000
   ```
3. Copy địa chỉ URL dạng `https://xxxx-xxx.ngrok-free.app` để dán vào cài đặt trên App di động.

---

## 4. Hướng dẫn chạy Mobile App Client (Flutter)

Ứng dụng Android viết bằng Flutter được cài đặt và vận hành trên điện thoại Android hoặc giả lập.

### Yêu cầu chuẩn bị:
* Cài đặt **Flutter SDK** và đảm bảo chạy lệnh `flutter doctor` đạt kết quả tốt (đặc biệt là mục Android toolchain).
* Bật chế độ **Gỡ lỗi USB (USB Debugging)** trên điện thoại Android và cắm cáp kết nối vào máy tính.

### Cách chạy:
1. Mở terminal tại thư mục dự án Flutter:
   ```powershell
   cd D:\Document\Code\DACN\RDDYOLOseg\mobile_app\flutter_app
   ```
2. Khởi tạo thư mục platform Android (chỉ cần chạy lần đầu tiên):
   ```powershell
   flutter create . --project-name rdd_mobile --org com.dacn.rdd --platforms android
   ```
3. Tải các package dependencies và chạy ứng dụng:
   ```powershell
   flutter pub get
   flutter run
   ```
4. If you want to build a release APK file to install directly on the phone:
   ```powershell
   flutter build apk --release
   ```
   * File APK kết quả sẽ nằm tại thư mục: `build/app/outputs/flutter-apk/app-release.apk`.

---

## 5. Sử dụng Ứng dụng Di động
1. Khởi động app di động, nhấn biểu tượng **Cài đặt (⚙️)** ở góc trên bên phải màn hình.
2. Dán địa chỉ URL của backend (ở **Mục 3**) vào và nhấn **Lưu và kiểm tra kết nối**.
3. Khi hiện kết nối thành công, quay về màn hình chính để thử nghiệm các chức năng:
   * **Chụp ảnh/Thư viện**: Chụp hình ảnh đường đi và gửi nhận diện, hỗ trợ lưu lịch sử và toạ độ GPS.
   * **Camera Realtime**: Stream video trực tiếp từ camera qua WebSocket lên server và hiển thị kết quả phân vùng tức thời.
   * **Upload Video**: Tải video và nhận kết quả video đã được xử lý phân vùng.
   * **Lịch sử & Bản đồ**: Xem lại các vết nứt, ổ gà kèm tọa độ GPS trên Google Maps.
