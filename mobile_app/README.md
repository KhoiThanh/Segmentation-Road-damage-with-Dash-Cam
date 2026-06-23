# Ứng dụng di động RDD - Nhận diện Ổ gà và Vết nứt

Ứng dụng Android (Flutter) + Backend (FastAPI) sử dụng model YOLOv11-seg đã được huấn luyện
trong dự án RDDYOLOseg để nhận diện và đếm số lượng **Ổ gà** và **Vết nứt** trên đường.

## Kiến trúc

```
   [ Điện thoại - Flutter App ]                     [ Máy tính - Backend ]
        |                                                  |
        |--- ảnh JPEG (multipart) ---POST /predict/image-->|
        |<--- ảnh kết quả + counts JSON --------------------|
        |                                                  |
        |--- video MP4 (multipart) ---POST /predict/video->|
        |<--- video kết quả + counts JSON ------------------|
        |                                                  |
        |--- WebSocket /ws/stream (frame JPEG) ----------->|--> YOLOv11-seg
        |<--- JPEG + counts realtime -----------------------|
        |                                                  |
        |--- GPS + counts + ảnh ---POST /history---------->|--> SQLite
        |<--- GET /history (list) --------------------------|
```

- Model `best.pt` chạy trên máy tính (có GPU/CPU mạnh hơn điện thoại).
- Điện thoại chỉ gửi ảnh/video/frame → nhận kết quả.
- Truy cập **bất kỳ đâu (4G/wifi khác mạng)** qua **ngrok** (free).

## Cấu trúc thư mục

```
mobile_app/
├── backend/
│   ├── main.py              # FastAPI server (image/video/websocket/history)
│   ├── requirements.txt
│   ├── run.bat              # Script Windows chạy server
│   ├── storage/             # Lưu ảnh/video kết quả (tự sinh)
│   └── history.db           # SQLite lịch sử (tự sinh)
└── flutter_app/
    ├── pubspec.yaml
    ├── lib/
    │   ├── main.dart
    │   ├── config.dart          # Quản lý URL backend
    │   ├── models/detection_result.dart
    │   ├── services/api_service.dart
    │   ├── services/location_service.dart
    │   ├── widgets/counts_card.dart
    │   └── screens/
    │       ├── home_screen.dart
    │       ├── capture_screen.dart    # Chụp ảnh / chọn từ thư viện
    │       ├── video_screen.dart      # Upload video
    │       ├── live_screen.dart       # Camera realtime qua WebSocket
    │       ├── history_screen.dart    # Lịch sử + GPS + mở Google Maps
    │       └── settings_screen.dart   # Cấu hình URL backend
    └── android_permissions/AndroidManifest_snippets.xml
```

---

## 1. Chạy Backend (trên máy tính có model)

> Lưu ý: PowerShell 5.1 (Windows) không hỗ trợ toán tử `&&`. Mỗi lệnh nên ở 1 dòng riêng,
> hoặc nối bằng `;` (chạy nối tiếp không dừng khi lỗi).

### Bước 1: Tạo môi trường ảo và cài requirements
```powershell
cd D:\Document\Code\DACN\RDDYOLOseg\mobile_app\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Nếu `Activate.ps1` báo lỗi execution policy, mở PowerShell **as Administrator** và chạy 1 lần:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
> Nếu đã có sẵn môi trường Python với `ultralytics` cho YoloSeg.ipynb thì có thể tái sử dụng,
> chỉ cần cài thêm: `pip install fastapi uvicorn[standard] python-multipart`.

### Bước 2: Chạy server
```powershell
# Cách 1: dùng script có sẵn
.\run.bat

# Cách 2: tự gọi uvicorn
$env:RDD_MODEL_PATH = "D:\Document\Code\DACN\RDDYOLOseg\runs\segment\Road_Patches_Cracks_Segmentation_Datasets_yolo11l\weights\best.pt"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
Server sẽ chạy ở `http://0.0.0.0:8000`. Mở trình duyệt thử `http://localhost:8000/health`
sẽ thấy `{"status":"ok",...}`.

### Bước 3: Mở ngrok để điện thoại truy cập từ 4G/mạng khác
1. Tải ngrok: https://ngrok.com/download (cài vào PATH).
2. Đăng ký lấy authtoken miễn phí và chạy: `ngrok config add-authtoken <TOKEN>`.
3. Mở terminal mới và chạy:
   ```powershell
   ngrok http 8000
   ```
4. Ngrok in ra URL dạng `https://abcd-xxx.ngrok-free.app` — copy URL này để dán vào app.

Nếu chỉ dùng **cùng wifi** với máy tính: dùng IP LAN của máy (chạy `ipconfig`, lấy IPv4),
ví dụ `http://192.168.1.10:8000`, không cần ngrok.

---

## 2. Build và chạy Flutter app

### Bước 1: Cài Flutter SDK
- Tải Flutter: https://docs.flutter.dev/get-started/install/windows
- Cài Android Studio (để có Android SDK + emulator).
- Kiểm tra: `flutter doctor` (xanh hết các mục Android toolchain).

### Bước 2: Tạo project và copy source

Vì project Flutter cần các thư mục Android/iOS được flutter tự sinh, làm theo (mỗi lệnh 1 dòng):

```powershell
cd D:\Document\Code\DACN\RDDYOLOseg\mobile_app\flutter_app
flutter create . --project-name rdd_mobile --org com.dacn.rdd --platforms android
flutter pub get
```
Lệnh `flutter create .` sẽ giữ nguyên `pubspec.yaml` và các file `lib/` hiện có,
chỉ tạo thêm thư mục `android/`, `ios/`, `test/`...

### Bước 3: Thêm permission Android
Mở file `android/app/src/main/AndroidManifest.xml` và **thêm các dòng `<uses-permission>` trước thẻ `<application>`** (xem file `android_permissions/AndroidManifest_snippets.xml`).

Đặc biệt:
- Nếu dùng **HTTP** (LAN, không phải ngrok https): thêm `android:usesCleartextTraffic="true"` vào thẻ `<application>`.
- Đặt `minSdkVersion` >= 21 trong `android/app/build.gradle` (mặc định Flutter đã ok).

### Bước 4: Cắm điện thoại Android (đã bật USB Debugging) hoặc dùng emulator
```powershell
flutter devices
flutter run
```

### Bước 5: Build APK để cài trực tiếp
```powershell
flutter build apk --release
```
File APK ở `build/app/outputs/flutter-apk/app-release.apk`. Copy sang điện thoại để cài.

---

## 3. Sử dụng app

1. Mở app, vào **Settings** (icon ⚙ ở góc phải app bar).
2. Dán URL backend: `https://abcd-xxx.ngrok-free.app` hoặc `http://192.168.1.10:8000`.
3. Nhấn **Lưu và kiểm tra** — nếu hiện "Kết nối thành công" là OK.
4. Quay lại Home, chọn 1 trong 4 chức năng:
   - **Chụp ảnh / Thư viện**: chụp 1 tấm hoặc chọn ảnh → tự gửi server → hiện ảnh đã vẽ mask + đếm. Nhấn **Lưu + Vị trí GPS** để lưu vào lịch sử kèm tọa độ.
   - **Camera Realtime**: nhấn START → app gửi frame liên tục qua WebSocket → hiển thị frame đã nhận diện với FPS thực tế.
   - **Upload Video**: chọn video → server xử lý từng frame → trả về MP4 đã vẽ + thống kê. Có thể chỉnh `sample_every` để tăng tốc (ví dụ chỉ infer 1/2 frame).
   - **Lịch sử + GPS**: xem lại các ảnh đã lưu kèm tọa độ; bấm **Maps** để mở vị trí trên Google Maps.

---

## 4. Mẹo hiệu năng

- Khi dùng ngrok free, có giới hạn băng thông và URL đổi mỗi lần restart — nên tự host nếu demo chính thức.
- Để livestream mượt: `live_screen.dart` đang gửi frame với cơ chế **back-pressure** (chỉ gửi frame mới khi server đã trả frame trước). Bạn có thể chỉnh độ phân giải bằng `ResolutionPreset.medium` → `low` để tăng FPS.
- Trên máy có GPU CUDA, model sẽ tự dùng GPU. Kiểm tra trong log uvicorn (`device=0` hoặc `device=cpu`).

## 5. API tham khảo

| Method | URL | Mô tả |
| --- | --- | --- |
| GET | `/health` | Kiểm tra server còn sống |
| GET | `/classes` | Danh sách class (en/vi) |
| POST | `/predict/image` | Form-data `file`. Trả ảnh kết quả + counts |
| POST | `/predict/video` | Form-data `file`, `sample_every`. Trả video MP4 + counts |
| WS | `/ws/stream` | Client gửi JPEG bytes → server gửi JSON `{image_b64, counts_vi, total}` |
| POST | `/history` | Form-data `counts_json`, `lat?`, `lng?`, `note?`, `image?` |
| GET | `/history` | Danh sách lịch sử (mới nhất trước) |
| DELETE | `/history/{id}` | Xóa 1 mục |
| GET | `/files/{name}` | Tải file ảnh/video kết quả |

## 6. Lưu ý khi nộp đồ án

- Có thể quay clip demo: bật backend trên máy → mở ngrok → cầm điện thoại đi quay đường → nhận diện realtime + lưu GPS.
- Bộ counts được tính sau **NMS** (đã loại bbox trùng) nên đếm khá chính xác trên ảnh tĩnh. Với video, mặc định cộng dồn theo frame nên 1 ổ gà xuất hiện trong 30 frame sẽ bị đếm 30 lần — nếu cần đếm "duy nhất theo object" thì phải bật tracking (ultralytics có `model.track(...)`); có thể bổ sung sau.
