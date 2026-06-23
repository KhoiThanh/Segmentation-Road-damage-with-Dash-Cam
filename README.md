# 🛣️ Road Damage Detection (RDD) - YOLO Segmentation

A comprehensive system for detecting and segmenting road damages (cracks, potholes) using YOLOv11 segmentation with three integrated components: Web application, Mobile backend API, and Flutter mobile app.
## [Dataset](https://www.kaggle.com/datasets/sunnykenneth/road-damage-dataset-for-segmentation)
## [Training Results](https://drive.google.com/drive/folders/1iMhYOiZlo6DkjC206jSGszwuBfUVISkh?usp=sharing)
## [Experimental Run Results](https://drive.google.com/drive/folders/163dhnXlHPX8bq6yulmcC1XvmP9MLlPmO?usp=sharing)
## [Mobile Application File](https://drive.google.com/drive/folders/1zmO5F4rPp-xRe5PBL1-W9EDodgj2jrSP?usp=sharing)

**Language:** English | [Tiếng Việt 🇻🇳](README_VI.md)

> 📖 For detailed Vietnamese guide: [Hướng dẫn Chi tiết](HUONG_DAN_CHAY.md)

---


## 🎯 Project Overview

This project provides an end-to-end solution for road damage detection and segmentation using state-of-the-art deep learning techniques. The system can:

- Detect and segment road damages (cracks, potholes) in real-time
- Process images, videos, and live camera streams
- Visualize model attention using Eigen-CAM and Grad-CAM heatmaps
- Track damage locations using GPS coordinates (mobile app)
- Provide cross-platform deployment (web and mobile)

### Use Cases

- **Road Infrastructure Management**: Monitor road conditions and schedule maintenance
- **Asset Management**: Maintain detailed records of road damage locations with GPS coordinates
- **Quality Assurance**: Automated inspection of road construction/repair quality
- **Insurance Claims**: Document road damage for insurance claims

---

## ✨ Features

### Web Application (Gradio)
- 📸 **Image Detection**: Upload images and get mask/bounding box predictions with confidence scores
- 🎥 **Video Detection**: Process videos frame-by-frame with automated damage detection
- 🔥 **Eigen-CAM Heatmaps**: Visualize which regions the model focuses on
- 📊 **Detection Statistics**: Summary of detected damages and confidence scores
- 🎨 **Interactive Interface**: User-friendly Gradio UI with drag-and-drop support

### Mobile Backend (FastAPI)
- ⚡ **Real-time Detection**: Process images and video frames in milliseconds
- 🔌 **WebSocket Support**: Live camera stream processing with WebSocket
- 💾 **Data Storage**: History tracking and GPS coordinate recording
- 🏥 **Health Check Endpoint**: Monitor server status and availability
- 📱 **RESTful API**: Clean API design for mobile integration
- 🔐 **CORS Configuration**: Secure cross-origin requests handling

### Mobile Application (Flutter)
- 📷 **Photo Capture & Library**: Take or select photos for detection
- 📹 **Live Camera Streaming**: Real-time video stream to backend via WebSocket
- 🎬 **Video Upload**: Process video files with detection results
- 📍 **GPS Integration**: Automatic GPS coordinate recording for damage locations
- 🗺️ **Google Maps Integration**: Visualize damage locations on interactive map
- 📱 **History & Records**: Browse previous detections with timestamps
- ⚙️ **Settings Panel**: Configure backend server connection

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model** | YOLOv11-seg | Segmentation & detection |
| **Backend ML** | PyTorch, Ultralytics | Deep learning framework |
| **Web Framework** | Gradio | Interactive web interface |
| **API Framework** | FastAPI | Backend REST/WebSocket API |
| **Mobile** | Flutter | Cross-platform mobile app |
| **Visualization** | OpenCV, PIL | Image processing and visualization |
| **Database** | File storage | History and GPS data |
| **Mapping** | Google Maps API | Location visualization |

---

## 📁 Project Structure

```
RDDYOLOseg/
├── README.md                              # This file
├── HUONG_DAN_CHAY.md                     # Vietnamese guide
├── web_app.py                            # Gradio web interface
├── model_optimization.py                 # Model optimization utilities
├── analyze_dataset.ipynb                 # Dataset analysis notebook
├── Dataset_Visualization_Segmentation_Features.ipynb  # Visualization notebook
├── Mask_RCNN_Training.ipynb             # Training notebook
├── YoloSeg.ipynb                        # YOLO segmentation notebook
├── yolo11n-seg.pt                       # Nano YOLO model weights
├── yolo26n.pt                           # Lightweight model weights
│
├── mobile_app/                          # Mobile application suite
│   ├── README.md                        # Mobile app documentation
│   ├── backend/                         # FastAPI backend server
│   │   ├── main.py                     # API application entry point
│   │   ├── requirements.txt            # Python dependencies
│   │   ├── run.bat                     # Windows batch runner
│   │   └── storage/                    # Data storage directory
│   │
│   └── flutter_app/                    # Flutter mobile application
│       ├── pubspec.yaml                # Flutter dependencies
│       ├── lib/
│       │   ├── main.dart              # App entry point
│       │   ├── config.dart            # Configuration constants
│       │   ├── models/                # Data models
│       │   ├── screens/               # UI screens
│       │   ├── services/              # API services
│       │   ├── utils/                 # Utility functions
│       │   └── widgets/               # Reusable widgets
│       ├── android/                   # Android-specific files
│       ├── assets/                    # App assets
│       └── test/                      # Test files
│
├── Road_Cracks_Segmentation_Datasets/  # Training dataset
│   ├── data.yaml                       # Dataset configuration
│   ├── train/                          # Training samples
│   ├── valid/                          # Validation samples
│   └── test/                           # Test samples
│
├── runs/                               # YOLO training outputs
│   └── segment/Road_Patches_Cracks_Segmentation_Datasets_yolo11l/
│
├── Test_Input/                         # Test input samples
│   ├── Input_test_image/              # Test images
│   └── Input_test_video/              # Test videos
│
└── Test_output/                        # Detection results
    ├── predictions_image/             # Image predictions
    ├── predictions_video/             # Video predictions
    ├── figures/                       # Visualization outputs
    └── image_detection_summary.csv    # Detection statistics
```

---

## 📦 Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.9 or higher (3.10+ recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: CUDA-capable GPU recommended for faster inference (NVIDIA RTX series preferred)
- **Storage**: 5GB+ for models, datasets, and outputs

### For Mobile App (Flutter)
- Flutter SDK (3.0+)
- Android SDK (API level 24+)
- Android Studio or VS Code with Flutter extension
- Android phone or emulator with USB Debugging enabled

---

## 🚀 Installation & Setup

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd RDDYOLOseg
```

### Step 2: Set Up Python Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install main dependencies
pip install gradio ultralytics opencv-python numpy torch torchvision pillow

# For web app specifically
pip install -r requirements.txt  # if exists

# For backend API
cd mobile_app/backend
pip install -r requirements.txt
cd ../..
```

### Step 4: Download/Prepare Models

The project includes pre-trained models:
- `yolo11n-seg.pt` - Nano model (lightweight, faster)
- `yolo26n.pt` - Custom lightweight model
- Best model in `runs/segment/Road_Patches_Cracks_Segmentation_Datasets_yolo11l/weights/best.pt`

Ensure these files are in the project root directory.

---

## ▶️ Running the Application

### Option 1: Web Application (Gradio)

```powershell
# From project root
python web_app.py

# Open browser to: http://127.0.0.1:7860
```

**Features:**
- Upload image or video files
- Real-time segmentation and bounding box detection
- Eigen-CAM heatmap visualization
- Download results

### Option 2: Mobile App Backend (FastAPI)

```powershell
# Navigate to backend
cd mobile_app/backend

# Setup virtual environment
python -m venv .venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run backend server
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Or use provided batch script:
.\run.bat

# API will be available at: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

### Option 3: Mobile App (Flutter)

```powershell
# Navigate to Flutter app
cd mobile_app/flutter_app

# Get dependencies
flutter pub get

# Run on connected device/emulator
flutter run

# Build APK for release
flutter build apk --release
```

**Note:** Configure backend server URL in app settings after installation.

---

## 🔗 API Documentation

### Backend Endpoints

#### Health Check
```
GET /health
Response: {"status": "healthy"}
```

#### Image Detection
```
POST /detect
Content-Type: multipart/form-data

Body:
  - file: <image_file>
  - confidence: <float> (0.0-1.0, default: 0.5)

Response: {
  "detections": [...],
  "inference_time": <float>,
  "image_with_boxes": <base64>
}
```

#### WebSocket Stream
```
WS /ws/stream
- Send: image frames as binary data
- Receive: detection results in real-time
```

Full API documentation available at `http://localhost:8000/docs` when backend is running.

---

## 📊 Dataset Information

### Road_Cracks_Segmentation_Datasets

**Source**: RoboFlow - Road Cracks Segmentation Dataset

**Dataset Statistics:**
- Total images: ~2000+
- Train/Val/Test split: 70/20/10 (approximately)
- Image format: PNG/JPG
- Annotation format: YOLO segmentation masks (.txt)

**Classes:**
- Longitudinal Cracks (Vết nứt dọc)
- Transverse Cracks (Vết nứt ngang)
- Pothole/Alligator Cracks (Ổ gà)

**Location:** `Road_Cracks_Segmentation_Datasets/`

**Configuration File:** `data.yaml`
```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images
nc: 1  # number of classes
names: ['road_damage']
```

---

## 🤖 Model Information

### YOLOv11 Segmentation

**Model Variants:**
- `yolo11n-seg.pt` - Nano (fastest, least accurate)
- `yolo26n.pt` - Custom lightweight variant
- `best.pt` - Full trained model (best accuracy)

**Training Details:**
```
Framework: YOLOv11 Ultralytics
Task: Instance Segmentation
Epochs: 100+
Batch Size: 16
Image Size: 640x640
Optimizer: SGD
Dataset: Road Cracks Segmentation Dataset
```

**Performance:**
- Inference Time: 50-100ms per frame (on GPU)
- Model Size: 20-50MB depending on variant

**Preprocessing:**
- Image normalization: [0, 1]
- Augmentation: Random rotation, flip, brightness, contrast
- Input resolution: 640x640

---

## 📈 Results & Outputs

### Web App Outputs

**Saved in:** `Test_output/`

```
Test_output/
├── predictions_image/      # Annotated images with masks
├── predictions_video/      # Video files with detection overlay
├── figures/                # Eigen-CAM visualizations
├── image_detection_summary.csv    # Statistics per image
└── video_detection_summary.csv    # Statistics per video
```

**Output CSV Format:**
```csv
filename,total_detections,avg_confidence,inference_time_ms,damage_type_1_count,...
image1.jpg,5,0.92,87.5,2,...
video1.mp4,156,0.88,45.2,78,...
```

### Mobile App Outputs

**Saved in:** `mobile_app/backend/storage/`

```
storage/
├── detections.json        # Detection history with GPS
├── images/                # Original uploaded images
└── results/               # Segmentation masks
```

**JSON Format:**
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

## 🔧 Troubleshooting

### Common Issues

#### 1. **TensorFlow/PyTorch Import Slow (20-30 seconds)**
**Solution:** This is normal on first load. Subsequent runs will be faster. Set environment variables:
```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
```

#### 2. **CUDA Not Found / GPU Not Detected**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, ensure NVIDIA drivers are installed:
# - Update GPU drivers from NVIDIA website
# - Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. **Unicode Characters Corrupted in Output (Vietnamese text)**
**Issue:** OpenCV `putText()` doesn't support Unicode characters properly
**Solution:** Use PIL (Pillow) for text rendering with TTF fonts, then convert back to OpenCV format

#### 4. **Backend Connection Issues (Mobile to Backend)**
**Steps:**
1. Ensure backend is running: `http://localhost:8000/health`
2. Get your PC's IP: `ipconfig` (look for IPv4 address, e.g., `192.168.1.15`)
3. Use `http://192.168.1.15:8000` in mobile app settings
4. Ensure firewall allows port 8000
5. For external access, use Ngrok: `ngrok http 8000`

#### 5. **Port 8000 Already in Use**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual ID)
taskkill /PID <PID> /F

# Or use different port
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

#### 6. **Flutter Build Errors**
```bash
# Clean and rebuild
flutter clean
flutter pub get
flutter pub upgrade
flutter run
```

#### 7. **Model File Not Found**
Ensure model files (.pt) are in the project root or update path in code:
```python
model = YOLO('path/to/model.pt')
```

---

## 📚 Notebooks

The project includes several Jupyter notebooks for exploration and training:

1. **YoloSeg.ipynb** - Main YOLO segmentation training and inference
2. **Mask_RCNN_Training.ipynb** - Alternative Mask R-CNN approach
3. **Dataset_Visualization_Segmentation_Features.ipynb** - Dataset exploration
4. **analyze_dataset.ipynb** - Statistical analysis
5. **model_optimization.py** - Model compression utilities

### Running Notebooks

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter server
jupyter notebook

# Or use VS Code with Jupyter extension
```

---

## 📝 Configuration

### Web App Settings
Modify `web_app.py`:
```python
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100
```

### Backend Settings
Modify `mobile_app/backend/main.py`:
```python
MODEL_PATH = "path/to/best.pt"
STORAGE_PATH = "./storage"
INFERENCE_TIMEOUT = 30  # seconds
```

### Mobile App Settings
Configure in app settings UI:
- Backend URL
- Model confidence threshold
- Auto-save detection history
- GPS tracking enable/disable

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes and commit (`git commit -m "Add improvement"`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

---

## 👥 Contributors

- **Developers**: [Add team member names]
- **Advisors**: [Add advisor names if applicable]
- **Dataset**: RoboFlow Community

---

## 📞 Support & Contact

For issues, questions, or suggestions:
- Open an Issue on GitHub
- Email: [contact email]
- Documentation: See [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md) for Vietnamese guide

---

## 🙏 Acknowledgments

- **YOLOv11**: Ultralytics for excellent segmentation models
- **Gradio**: Simple interactive interfaces
- **FastAPI**: Modern and fast web framework
- **Flutter**: Cross-platform mobile development
- **Dataset**: RoboFlow for road damage segmentation dataset

---

## 📋 Changelog

### Version 1.0.0 (2024-06-23)
- Initial release with three main components
- Web app with Gradio interface
- FastAPI backend with WebSocket support
- Flutter mobile application
- Full documentation and examples

---

**Last Updated:** 2024-06-23  
**Status:** Active Development  
**Python Version:** 3.9+  
**YOLO Version:** v11
