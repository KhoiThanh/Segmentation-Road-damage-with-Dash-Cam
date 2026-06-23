# 🛣️ Hệ thống Nhận diện Tổn thương Đường (RDD) - YOLO Segmentation
## Dataset "https://www.kaggle.com/datasets/sunnykenneth/road-damage-dataset-for-segmentation"
## Kết quả quá trình huấn luyện : "https://drive.google.com/drive/folders/1iMhYOiZlo6DkjC206jSGszwuBfUVISkh?usp=sharing"
## Kết quả chạy thực nghiệm : "https://drive.google.com/drive/folders/163dhnXlHPX8bq6yulmcC1XvmP9MLlPmO?usp=sharing"
## Tệp ứng dụng mobile: "https://drive.google.com/drive/folders/1zmO5F4rPp-xRe5PBL1-W9EDodgj2jrSP?usp=sharing"
Một hệ thống toàn diện để phát hiện và phân vùng các tổn thương đường bộ (vết nứt, ổ gà) sử dụng phân vùng YOLOv11 với ba thành phần tích hợp: Ứng dụng Web, API Backend di động và ứng dụng Flutter.

**Ngôn ngữ:** [English](README.md) | Tiếng Việt 🇻🇳

---


## 🎯 Tổng quan Dự án

Dự án này cung cấp giải pháp toàn diện cho việc phát hiện và phân vùng tổn thương đường bộ sử dụng các kỹ thuật học sâu hiện đại. Hệ thống có thể:

- Phát hiện và phân vùng tổn thương đường (vết nứt, ổ gà) trong thời gian thực
- Xử lý ảnh, video và luồng camera trực tiếp
- Trực quan hóa sự tập trung của mô hình sử dụng Eigen-CAM và Grad-CAM
- Theo dõi vị trí tổn thương bằng tọa độ GPS (ứng dụng di động)
- Cung cấp triển khai đa nền tảng (web và di động)

### Các ứng dụng

- **Quản lý Cơ sở hạ tầng Đường bộ**: Giám sát tình trạng đường và lập kế hoạch bảo trì
- **Quản lý Tài sản**: Duy trì hồ sơ chi tiết về vị trí tổn thương với tọa độ GPS
- **Đảm bảo Chất lượng**: Kiểm tra tự động chất lượng xây dựng/sửa chữa đường
- **Yêu cầu Bảo hiểm**: Ghi chép tổn thương đường cho các yêu cầu bảo hiểm

---

## ✨ Tính năng

### Ứng dụng Web (Gradio)
- 📸 **Phát hiện Hình ảnh**: Tải lên ảnh và nhận kết quả mask/bounding box với điểm tin cậy
- 🎥 **Phát hiện Video**: Xử lý video theo từng frame với phát hiện tổn thương tự động
- 🔥 **Eigen-CAM Heatmaps**: Trực quan hóa các vùng mà mô hình tập trung chú ý
- 📊 **Thống kê Phát hiện**: Tóm tắt các tổn thương được phát hiện và điểm tin cậy
- 🎨 **Giao diện Tương tác**: Giao diện Gradio thân thiện với hỗ trợ kéo thả

### Backend di động (FastAPI)
- ⚡ **Phát hiện Thời gian thực**: Xử lý ảnh và video frame trong vòng millisecond
- 🔌 **Hỗ trợ WebSocket**: Xử lý luồng camera trực tiếp qua WebSocket
- 💾 **Lưu trữ Dữ liệu**: Theo dõi lịch sử và ghi chép tọa độ GPS
- 🏥 **Endpoint Kiểm tra Sức khỏe**: Giám sát trạng thái và tính khả dụng của máy chủ
- 📱 **API RESTful**: Thiết kế API rõ ràng cho tích hợp di động
- 🔐 **Xử lý CORS**: Xử lý yêu cầu liên nguồn gốc an toàn

### Ứng dụng Di động (Flutter)
- 📷 **Chụp Ảnh & Thư viện**: Chụp hoặc chọn ảnh để phát hiện
- 📹 **Truyền phát Camera Trực tiếp**: Truyền video trực tiếp đến backend qua WebSocket
- 🎬 **Tải lên Video**: Xử lý tập tin video với kết quả phát hiện
- 📍 **Tích hợp GPS**: Ghi chép tự động tọa độ GPS cho các vị trí tổn thương
- 🗺️ **Tích hợp Google Maps**: Trực quan hóa vị trí tổn thương trên bản đồ tương tác
- 📱 **Lịch sử & Hồ sơ**: Duyệt qua các phát hiện trước đó với dấu thời gian
- ⚙️ **Bảng điều khiển Cài đặt**: Cấu hình kết nối máy chủ backend

---

## 🛠️ Stack Công nghệ

| Thành phần | Công nghệ | Mục đích |
|-----------|-----------|---------|
| **Mô hình** | YOLOv11-seg | Phân vùng & phát hiện |
| **Backend ML** | PyTorch, Ultralytics | Khung học sâu |
| **Framework Web** | Gradio | Giao diện web tương tác |
| **Framework API** | FastAPI | API REST/WebSocket backend |
| **Di động** | Flutter | Ứng dụng di động đa nền tảng |
| **Trực quan hóa** | OpenCV, PIL | Xử lý và trực quan hóa ảnh |
| **Cơ sở dữ liệu** | Lưu trữ File | Lịch sử và dữ liệu GPS |
| **Lập bản đồ** | Google Maps API | Trực quan hóa vị trí |

---

## 📁 Cấu trúc Dự án

```
RDDYOLOseg/
├── README.md                              # File English (chính)
├── README_VI.md                           # File này (tiếng Việt)
├── HUONG_DAN_CHAY.md                     # Hướng dẫn chi tiết tiếng Việt
├── web_app.py                            # Giao diện web Gradio
├── model_optimization.py                 # Tiện ích tối ưu hóa mô hình
├── analyze_dataset.ipynb                 # Notebook phân tích dataset
├── Dataset_Visualization_Segmentation_Features.ipynb  # Notebook trực quan hóa
├── Mask_RCNN_Training.ipynb             # Notebook huấn luyện
├── YoloSeg.ipynb                        # Notebook YOLO segmentation
├── yolo11n-seg.pt                       # Trọng số mô hình Nano
├── yolo26n.pt                           # Trọng số mô hình nhẹ
│
├── mobile_app/                          # Bộ ứng dụng di động
│   ├── README.md                        # Tài liệu ứng dụng di động
│   ├── backend/                         # Máy chủ backend FastAPI
│   │   ├── main.py                     # Điểm vào ứng dụng API
│   │   ├── requirements.txt            # Phụ thuộc Python
│   │   ├── run.bat                     # Trình chạy batch Windows
│   │   └── storage/                    # Thư mục lưu trữ dữ liệu
│   │
│   └── flutter_app/                    # Ứng dụng di động Flutter
│       ├── pubspec.yaml                # Phụ thuộc Flutter
│       ├── lib/
│       │   ├── main.dart              # Điểm vào ứng dụng
│       │   ├── config.dart            # Hằng số cấu hình
│       │   ├── models/                # Mô hình dữ liệu
│       │   ├── screens/               # Các màn hình UI
│       │   ├── services/              # Dịch vụ API
│       │   ├── utils/                 # Hàm tiện ích
│       │   └── widgets/               # Widget tái sử dụng
│       ├── android/                   # Tập tin Android
│       ├── assets/                    # Tài sản ứng dụng
│       └── test/                      # Tập tin kiểm tra
│
├── Road_Cracks_Segmentation_Datasets/  # Dataset huấn luyện
│   ├── data.yaml                       # Cấu hình dataset
│   ├── train/                          # Mẫu huấn luyện
│   ├── valid/                          # Mẫu xác thực
│   └── test/                           # Mẫu kiểm tra
│
├── runs/                               # Output huấn luyện YOLO
│   └── segment/Road_Patches_Cracks_Segmentation_Datasets_yolo11l/
│
├── Test_Input/                         # Mẫu đầu vào kiểm tra
│   ├── Input_test_image/              # Ảnh kiểm tra
│   └── Input_test_video/              # Video kiểm tra
│
└── Test_output/                        # Kết quả phát hiện
    ├── predictions_image/             # Dự đoán ảnh
    ├── predictions_video/             # Dự đoán video
    ├── figures/                       # Output trực quan hóa
    └── image_detection_summary.csv    # Thống kê phát hiện
```

---

## 📦 Yêu cầu Hệ thống

### Yêu cầu Hệ điều hành
- **OS**: Windows 10/11, Linux, hoặc macOS
- **Python**: 3.9 trở lên (khuyến khích 3.10+)
- **RAM**: Tối thiểu 8GB (16GB khuyến khích)
- **GPU**: GPU có hỗ trợ CUDA khuyến khích để suy luận nhanh hơn (NVIDIA RTX ưu tiên)
- **Lưu trữ**: 5GB+ cho mô hình, dataset và output

### Cho Ứng dụng Di động (Flutter)
- Flutter SDK (3.0+)
- Android SDK (API level 24+)
- Android Studio hoặc VS Code với phần mở rộng Flutter
- Điện thoại Android hoặc giả lập với Gỡ lỗi USB bật

---

## 🚀 Cài đặt & Thiết lập

### Bước 1: Clone Repository

```bash
git clone <repository-url>
cd RDDYOLOseg
```

### Bước 2: Thiết lập Môi trường Python

```powershell
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
.\venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source venv/bin/activate
```

### Bước 3: Cài đặt Phụ thuộc

```bash
# Cài đặt phụ thuộc chính
pip install gradio ultralytics opencv-python numpy torch torchvision pillow

# Cho ứng dụng web cụ thể
pip install -r requirements.txt  # nếu tồn tại

# Cho backend API
cd mobile_app/backend
pip install -r requirements.txt
cd ../..
```

### Bước 4: Tải xuống/Chuẩn bị Mô hình

Dự án bao gồm các mô hình được huấn luyện trước:
- `yolo11n-seg.pt` - Mô hình Nano (nhẹ, nhanh hơn)
- `yolo26n.pt` - Mô hình nhẹ tùy chỉnh
- Mô hình tốt nhất trong `runs/segment/Road_Patches_Cracks_Segmentation_Datasets_yolo11l/weights/best.pt`

Đảm bảo các tập tin này nằm trong thư mục gốc dự án.

---

## ▶️ Chạy Ứng dụng

### Tùy chọn 1: Ứng dụng Web (Gradio)

```powershell
# Từ thư mục gốc dự án
python web_app.py

# Mở trình duyệt: http://127.0.0.1:7860
```

**Tính năng:**
- Tải lên tập tin ảnh hoặc video
- Phân vùng và phát hiện bounding box thời gian thực
- Trực quan hóa Eigen-CAM heatmap
- Tải xuống kết quả

### Tùy chọn 2: Backend Ứng dụng Di động (FastAPI)

```powershell
# Điều hướng đến backend
cd mobile_app/backend

# Thiết lập môi trường ảo
python -m venv .venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Chạy máy chủ backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Hoặc sử dụng script batch đã chuẩn bị:
.\run.bat

# API sẽ có sẵn tại: http://localhost:8000
# Tài liệu: http://localhost:8000/docs
```

**Kiểm tra Sức khỏe:**
```bash
curl http://localhost:8000/health
```

### Tùy chọn 3: Ứng dụng Di động (Flutter)

```powershell
# Điều hướng đến ứng dụng Flutter
cd mobile_app/flutter_app

# Lấy phụ thuộc
flutter pub get

# Chạy trên thiết bị/giả lập được kết nối
flutter run

# Xây dựng APK cho phiên bản phát hành
flutter build apk --release
```

**Lưu ý:** Cấu hình URL máy chủ backend trong cài đặt ứng dụng sau khi cài đặt.

---

## 🔗 Tài liệu API

### Các Endpoint Backend

#### Kiểm tra Sức khỏe
```
GET /health
Phản hồi: {"status": "healthy"}
```

#### Phát hiện Hình ảnh
```
POST /detect
Content-Type: multipart/form-data

Thân:
  - file: <image_file>
  - confidence: <float> (0.0-1.0, mặc định: 0.5)

Phản hồi: {
  "detections": [...],
  "inference_time": <float>,
  "image_with_boxes": <base64>
}
```

#### Luồng WebSocket
```
WS /ws/stream
- Gửi: các frame ảnh dưới dạng dữ liệu nhị phân
- Nhận: kết quả phát hiện thời gian thực
```

Tài liệu API đầy đủ có sẵn tại `http://localhost:8000/docs` khi backend đang chạy.

---

## 📊 Thông tin Dataset

### Road_Cracks_Segmentation_Datasets

**Nguồn**: RoboFlow - Road Cracks Segmentation Dataset

**Thống kê Dataset:**
- Tổng ảnh: ~2000+
- Chia train/val/test: 70/20/10 (xấp xỉ)
- Định dạng ảnh: PNG/JPG
- Định dạng chú thích: YOLO segmentation masks (.txt)

**Các lớp:**
- Vết nứt Dọc (Longitudinal Cracks)
- Vết nứt Ngang (Transverse Cracks)
- Ổ gà/Vết nứt Cá sấu (Pothole/Alligator Cracks)

**Vị trí:** `Road_Cracks_Segmentation_Datasets/`

**Tệp Cấu hình:** `data.yaml`
```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images
nc: 1  # số lớp
names: ['road_damage']
```

---

## 🤖 Thông tin Mô hình

### Phân vùng YOLOv11

**Các phiên bản Mô hình:**
- `yolo11n-seg.pt` - Nano (nhanh nhất, chính xác nhất)
- `yolo26n.pt` - Biến thể nhẹ tùy chỉnh
- `best.pt` - Mô hình được huấn luyện đầy đủ (độ chính xác tốt nhất)

**Chi tiết Huấn luyện:**
```
Framework: YOLOv11 Ultralytics
Tác vụ: Phân vùng Thể hiện
Epochs: 100+
Kích thước Batch: 16
Kích thước Ảnh: 640x640
Trình tối ưu hóa: SGD
Dataset: Road Cracks Segmentation Dataset
```

**Hiệu suất:**
- Thời gian Suy luận: 50-100ms mỗi frame (trên GPU)
- Kích thước Mô hình: 20-50MB tùy theo biến thể

**Tiền xử lý:**
- Chuẩn hóa ảnh: [0, 1]
- Tăng cường: Xoay ngẫu nhiên, lật, độ sáng, độ tương phản
- Độ phân giải Đầu vào: 640x640

---

## 📈 Kết quả & Output

### Output Ứng dụng Web

**Được lưu trong:** `Test_output/`

```
Test_output/
├── predictions_image/      # Ảnh chú thích với mask
├── predictions_video/      # Tập tin video với overlay phát hiện
├── figures/                # Trực quan hóa Eigen-CAM
├── image_detection_summary.csv    # Thống kê trên mỗi ảnh
└── video_detection_summary.csv    # Thống kê trên mỗi video
```

**Định dạng CSV Output:**
```csv
filename,total_detections,avg_confidence,inference_time_ms,damage_type_1_count,...
image1.jpg,5,0.92,87.5,2,...
video1.mp4,156,0.88,45.2,78,...
```

### Output Ứng dụng Di động

**Được lưu trong:** `mobile_app/backend/storage/`

```
storage/
├── detections.json        # Lịch sử phát hiện với GPS
├── images/                # Ảnh tải lên gốc
└── results/               # Mask phân vùng
```

**Định dạng JSON:**
```json
{
  "id": "20240623_153045",
  "timestamp": "2024-06-23T15:30:45Z",
  "latitude": 10.7769,
  "longitude": 106.7009,
  "damage_type": "pothole",
  "confidence": 0.94,
  "area_pixels": 2856,
  "severity": "high"
}
```

---

## 🔧 Khắc phục Sự cố

### Các Vấn đề Phổ biến

#### 1. **TensorFlow/PyTorch Import Chậm (20-30 giây)**
**Giải pháp:** Điều này bình thường khi tải lần đầu. Các lần chạy tiếp theo sẽ nhanh hơn. Đặt các biến môi trường:
```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
```

#### 2. **CUDA Không Tìm thấy / GPU Không Được Phát hiện**
```bash
# Kiểm tra tính khả dụng CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu False, đảm bảo trình điều khiển NVIDIA được cài đặt:
# - Cập nhật trình điều khiển GPU từ trang web NVIDIA
# - Cài đặt lại PyTorch với phiên bản CUDA chính xác
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. **Các Ký tự Unicode Bị Hỏng trong Output (Văn bản Tiếng Việt)**
**Vấn đề:** OpenCV `putText()` không hỗ trợ ký tự Unicode đúng cách
**Giải pháp:** Sử dụng PIL (Pillow) để hiển thị văn bản với phông chữ TTF, sau đó chuyển đổi trở lại định dạng OpenCV

#### 4. **Vấn đề Kết nối Backend (Di động đến Backend)**
**Các bước:**
1. Đảm bảo backend đang chạy: `http://localhost:8000/health`
2. Lấy IP của PC: `ipconfig` (tìm địa chỉ IPv4, ví dụ: `192.168.1.15`)
3. Sử dụng `http://192.168.1.15:8000` trong cài đặt ứng dụng di động
4. Đảm bảo tường lửa cho phép cổng 8000
5. Để truy cập bên ngoài, sử dụng Ngrok: `ngrok http 8000`

#### 5. **Cổng 8000 Đã Được Sử dụng**
```powershell
# Tìm quy trình sử dụng cổng 8000
netstat -ano | findstr :8000

# Giết quy trình (thay PID bằng ID thực tế)
taskkill /PID <PID> /F

# Hoặc sử dụng cổng khác
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

#### 6. **Lỗi Xây dựng Flutter**
```bash
# Làm sạch và xây dựng lại
flutter clean
flutter pub get
flutter pub upgrade
flutter run
```

#### 7. **Tệp Mô hình Không Tìm thấy**
Đảm bảo các tệp mô hình (.pt) nằm trong thư mục gốc dự án hoặc cập nhật đường dẫn trong mã:
```python
model = YOLO('path/to/model.pt')
```

---

## 📚 Các Notebook

Dự án bao gồm một số Jupyter notebook cho khám phá và huấn luyện:

1. **YoloSeg.ipynb** - Huấn luyện và suy luận phân vùng YOLO chính
2. **Mask_RCNN_Training.ipynb** - Phương pháp Mask R-CNN thay thế
3. **Dataset_Visualization_Segmentation_Features.ipynb** - Khám phá dataset
4. **analyze_dataset.ipynb** - Phân tích thống kê
5. **model_optimization.py** - Tiện ích nén mô hình

### Chạy Notebook

```bash
# Cài đặt Jupyter
pip install jupyter

# Khởi động máy chủ Jupyter
jupyter notebook

# Hoặc sử dụng VS Code với phần mở rộng Jupyter
```

---

## 📝 Cấu hình

### Cài đặt Ứng dụng Web
Sửa đổi `web_app.py`:
```python
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100
```

### Cài đặt Backend
Sửa đổi `mobile_app/backend/main.py`:
```python
MODEL_PATH = "path/to/best.pt"
STORAGE_PATH = "./storage"
INFERENCE_TIMEOUT = 30  # giây
```

### Cài đặt Ứng dụng Di động
Cấu hình trong giao diện cài đặt ứng dụng:
- URL Backend
- Ngưỡng tin cậy mô hình
- Lưu tự động lịch sử phát hiện
- Bật/tắt theo dõi GPS

---

## 🤝 Đóng góp

Chúng tôi hoan nghênh các đóng góp! Vui lòng tuân theo các hướng dẫn sau:

1. Fork repository
2. Tạo nhánh tính năng (`git checkout -b feature/improvement`)
3. Thực hiện thay đổi và commit (`git commit -m "Add improvement"`)
4. Push đến nhánh (`git push origin feature/improvement`)
5. Mở Pull Request

---

## 📄 Giấy phép

[Chỉ định giấy phép của bạn tại đây - ví dụ: MIT, Apache 2.0, v.v.]

---

## 👥 Những người đóng góp

- **Nhà phát triển**: [Thêm tên thành viên nhóm]
- **Cố vấn**: [Thêm tên cố vấn nếu có]
- **Dataset**: RoboFlow Community

---

## 📞 Hỗ trợ & Liên hệ

Để báo cáo vấn đề, câu hỏi hoặc đề xuất:
- Mở một Issue trên GitHub
- Email: [địa chỉ email liên hệ]
- Tài liệu: Xem [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md) để biết hướng dẫn tiếng Việt chi tiết

---

## 🙏 Lời cảm ơn

- **YOLOv11**: Ultralytics cung cấp các mô hình phân vùng xuất sắc
- **Gradio**: Giao diện tương tác đơn giản
- **FastAPI**: Framework web hiện đại và nhanh chóng
- **Flutter**: Phát triển ứng dụng di động đa nền tảng
- **Dataset**: RoboFlow cung cấp dataset phân vùng tổn thương đường

---

## 📋 Nhật ký Thay đổi

### Phiên bản 1.0.0 (2024-06-23)
- Phát hành ban đầu với ba thành phần chính
- Ứng dụng web với giao diện Gradio
- Backend FastAPI với hỗ trợ WebSocket
- Ứng dụng di động Flutter
- Tài liệu đầy đủ và ví dụ

---

**Cập nhật Lần cuối:** 2024-06-23  
**Trạng thái:** Phát triển Tích cực  
**Phiên bản Python:** 3.9+  
**Phiên bản YOLO:** v11
