"""
RDD YOLO Segmentation - FastAPI backend.

Endpoints:
- GET  /health                 : kiem tra ket noi
- GET  /classes                : danh sach class
- POST /predict/image          : nhan dien anh, tra ve anh ket qua va counts
- POST /predict/video          : nhan dien video, tra ve mp4 ket qua va counts
- WS   /ws/stream              : realtime, client gui JPEG -> tra ve JPEG + counts
- POST /history                : luu lich su (json: lat, lng, counts, note, image_url...)
- GET  /history                : danh sach lich su
- DELETE /history/{id}         : xoa
- GET  /files/{name}           : static file (anh / video da nhan dien)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import tempfile
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Cau hinh
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get(
    "RDD_MODEL_PATH",
    r"D:\Document\Code\DACN\RDDYOLOseg\runs\segment\Road_Patches_Cracks_Segmentation_Datasets_yolo11l\weights\best.pt",
)
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / "history.db"

CONF_THRESHOLD_DEFAULT = 0.2
IOU_THRESHOLD_DEFAULT = 0.45

# Class name tieng Viet (map tu data.yaml: 0=Crack, 1=Pothole)
VI_NAMES = {"Crack": "Vet nut", "Pothole": "O ga"}


# ----------------------------------------------------------------------------
# Model & DB
# ----------------------------------------------------------------------------
model: YOLO | None = None


def get_model() -> YOLO:
    global model
    if model is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(f"Khong tim thay model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
    return model


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            lat REAL,
            lng REAL,
            note TEXT,
            counts_json TEXT NOT NULL,
            image_file TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _names_dict(result: Any) -> dict[int, str]:
    names = getattr(result, "names", None) or getattr(get_model(), "names", {})
    return names if isinstance(names, dict) else {}


def _count_classes(result: Any) -> dict[str, int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return {}
    try:
        cls_list = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    except Exception:
        try:
            cls_list = [int(x) for x in boxes.cls]
        except Exception:
            cls_list = []
    if not cls_list:
        return {}
    names = _names_dict(result)
    counts = Counter(cls_list)
    out: dict[str, int] = {}
    for cls_id, n in counts.items():
        en_name = names.get(cls_id, str(cls_id))
        out[en_name] = out.get(en_name, 0) + n
    return out


def _to_vi_counts(counts: dict[str, int]) -> dict[str, int]:
    return {VI_NAMES.get(k, k): v for k, v in counts.items()}


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Khong doc duoc anh")
    return img


def _save_image_jpg(image: np.ndarray, name_prefix: str = "img") -> str:
    fname = f"{name_prefix}_{uuid.uuid4().hex}.jpg"
    fpath = STORAGE_DIR / fname
    cv2.imwrite(str(fpath), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return fname


# ----------------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    # Warm up model
    try:
        get_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        get_model()(dummy, conf=CONF_THRESHOLD_DEFAULT, iou=IOU_THRESHOLD_DEFAULT, verbose=False)
        print("Model loaded & warmed up.")
    except Exception as e:
        print(f"Loi load model: {e}")
    yield


app = FastAPI(title="RDD YOLO Seg Mobile API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": Path(MODEL_PATH).name, "time": time.time()}


@app.get("/classes")
def classes() -> dict[str, Any]:
    names = getattr(get_model(), "names", {}) or {}
    return {
        "en": names,
        "vi": {k: VI_NAMES.get(v, v) for k, v in names.items()} if isinstance(names, dict) else {},
    }


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Form(CONF_THRESHOLD_DEFAULT),
    iou: float = Form(IOU_THRESHOLD_DEFAULT),
    roi_enabled: bool = Form(False),
    roi_top: int = Form(0),
    roi_bottom: int = Form(100),
    roi_left: int = Form(0),
    roi_right: int = Form(100),
) -> JSONResponse:
    data = await file.read()
    img = _decode_image(data)

    results = get_model()(img, conf=conf, iou=iou, verbose=False)
    res = results[0]
    
    # Áp dụng ROI nếu được kích hoạt và hợp lệ
    has_valid_roi = roi_enabled and (roi_top < roi_bottom) and (roi_left < roi_right)
    if has_valid_roi:
        height, width = img.shape[:2]
        ymin = int(height * roi_top / 100)
        ymax = int(height * roi_bottom / 100)
        xmin = int(width * roi_left / 100)
        xmax = int(width * roi_right / 100)
        
        keep_indices = []
        boxes = getattr(res, "boxes", None)
        if boxes is not None and boxes.xyxy is not None:
            try:
                xyxy = boxes.xyxy.detach().cpu().numpy()
            except Exception:
                xyxy = boxes.xyxy
            for i, box in enumerate(xyxy):
                bx1, by1, bx2, by2 = box[:4]
                bcx = (bx1 + bx2) / 2
                bcy = (by1 + by2) / 2
                if xmin <= bcx <= xmax and ymin <= bcy <= ymax:
                    keep_indices.append(i)
        res = res[keep_indices]

    plotted = res.plot()  # BGR ndarray with masks/boxes
    
    # Vẽ khung ROI lên kết quả
    if has_valid_roi:
        cv2.rectangle(plotted, (xmin, ymin), (xmax, ymax), (46, 204, 113), 2)
        cv2.putText(plotted, "VUNG ROI", (xmin + 10, ymin + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113), 2, cv2.LINE_AA)

    en_counts = _count_classes(res)
    vi_counts = _to_vi_counts(en_counts)
    total = int(sum(en_counts.values()))

    out_name = _save_image_jpg(plotted, name_prefix="det")
    return JSONResponse(
        {
            "image_url": f"/files/{out_name}",
            "image_file": out_name,
            "total": total,
            "counts_en": en_counts,
            "counts_vi": vi_counts,
        }
    )


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    conf: float = Form(CONF_THRESHOLD_DEFAULT),
    iou: float = Form(IOU_THRESHOLD_DEFAULT),
    sample_every: int = Form(1),
    roi_enabled: bool = Form(False),
    roi_top: int = Form(0),
    roi_bottom: int = Form(100),
    roi_left: int = Form(0),
    roi_right: int = Form(100),
) -> JSONResponse:
    """sample_every: chi infer moi N frame (de tang toc). >=1."""
    sample_every = max(1, int(sample_every))

    # Luu file tam vao storage de OpenCV doc duoc
    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    tmp_in_fd, tmp_in_path = tempfile.mkstemp(suffix=suffix, dir=STORAGE_DIR)
    os.close(tmp_in_fd)
    with open(tmp_in_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(tmp_in_path)
    if not cap.isOpened():
        os.unlink(tmp_in_path)
        raise HTTPException(status_code=400, detail="Khong mo duoc video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_name = f"vid_{uuid.uuid4().hex}.mp4"
    out_path = STORAGE_DIR / out_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frames = 0
    frames_with_detections = 0
    cumulative: Counter[int] = Counter()
    last_plotted = None
    mdl = get_model()
    names = getattr(mdl, "names", {}) or {}

    # Tính toán tọa độ ROI nếu bật
    has_valid_roi = roi_enabled and (roi_top < roi_bottom) and (roi_left < roi_right)
    if has_valid_roi:
        ymin = int(height * roi_top / 100)
        ymax = int(height * roi_bottom / 100)
        xmin = int(width * roi_left / 100)
        xmax = int(width * roi_right / 100)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        do_infer = (frames % sample_every) == 1 or sample_every == 1
        if do_infer:
            results = mdl(frame, conf=conf, iou=iou, verbose=False)
            res = results[0]
            
            if has_valid_roi:
                keep_indices = []
                boxes = getattr(res, "boxes", None)
                if boxes is not None and boxes.xyxy is not None:
                    try:
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                    except Exception:
                        xyxy = boxes.xyxy
                    for i, box in enumerate(xyxy):
                        bx1, by1, bx2, by2 = box[:4]
                        bcx = (bx1 + bx2) / 2
                        bcy = (by1 + by2) / 2
                        if xmin <= bcx <= xmax and ymin <= bcy <= ymax:
                            keep_indices.append(i)
                res = res[keep_indices]
                
            last_plotted = res.plot()
            
            if has_valid_roi:
                cv2.rectangle(last_plotted, (xmin, ymin), (xmax, ymax), (46, 204, 113), 2)
                cv2.putText(last_plotted, "VUNG ROI", (xmin + 10, ymin + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113), 2, cv2.LINE_AA)

            boxes = getattr(res, "boxes", None)
            if boxes is not None and boxes.cls is not None:
                try:
                    cls_list = boxes.cls.detach().cpu().numpy().astype(int).tolist()
                except Exception:
                    cls_list = []
                if cls_list:
                    frames_with_detections += 1
                    cumulative.update(cls_list)
            writer.write(last_plotted)
        else:
            # tai su dung anh da ve cua frame gan nhat de giu nhip video
            writer.write(last_plotted if last_plotted is not None else frame)

    cap.release()
    writer.release()
    try:
        os.unlink(tmp_in_path)
    except OSError:
        pass

    en_counts: dict[str, int] = {}
    for cls_id, n in cumulative.items():
        en_counts[names.get(cls_id, str(cls_id))] = int(n)
    vi_counts = _to_vi_counts(en_counts)

    return JSONResponse(
        {
            "video_url": f"/files/{out_name}",
            "video_file": out_name,
            "frames": frames,
            "frames_with_detections": frames_with_detections,
            "total": int(sum(en_counts.values())),
            "counts_en": en_counts,
            "counts_vi": vi_counts,
        }
    )


@app.websocket("/ws/stream")
async def ws_stream(
    ws: WebSocket,
    roi_enabled: bool = False,
    roi_top: int = 0,
    roi_bottom: int = 100,
    roi_left: int = 0,
    roi_right: int = 100,
) -> None:
    """
    Client gui frame JPEG (bytes binary), server tra ve JSON:
        { "image_b64": "...", "counts_vi": {...}, "total": N, "ts": ... }
    """
    await ws.accept()
    mdl = get_model()
    try:
        while True:
            data = await ws.receive_bytes()
            if not data:
                continue

            # Kiểm tra tương thích ngược: JPEG bắt đầu bằng FF D8 (255, 216)
            if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
                seq = 0
                jpeg_data = data
            else:
                if len(data) < 4:
                    continue
                seq = int.from_bytes(data[:4], byteorder='big')
                jpeg_data = data[4:]

            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Inference (chay trong thread executor de khong block event loop)
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None,
                lambda f=frame: mdl(f, conf=CONF_THRESHOLD_DEFAULT, iou=IOU_THRESHOLD_DEFAULT, verbose=False),
            )
            res = results[0]
            
            # Áp dụng ROI nếu được kích hoạt và hợp lệ
            has_valid_roi = roi_enabled and (roi_top < roi_bottom) and (roi_left < roi_right)
            if has_valid_roi:
                height, width = frame.shape[:2]
                ymin = int(height * roi_top / 100)
                ymax = int(height * roi_bottom / 100)
                xmin = int(width * roi_left / 100)
                xmax = int(width * roi_right / 100)
                
                keep_indices = []
                boxes = getattr(res, "boxes", None)
                if boxes is not None and boxes.xyxy is not None:
                    try:
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                    except Exception:
                        xyxy = boxes.xyxy
                    for i, box in enumerate(xyxy):
                        bx1, by1, bx2, by2 = box[:4]
                        bcx = (bx1 + bx2) / 2
                        bcy = (by1 + by2) / 2
                        if xmin <= bcx <= xmax and ymin <= bcy <= ymax:
                            keep_indices.append(i)
                res = res[keep_indices]

            plotted = res.plot()
            
            # Vẽ viền ROI lên khung hình stream
            if has_valid_roi:
                cv2.rectangle(plotted, (xmin, ymin), (xmax, ymax), (46, 204, 113), 2)
                cv2.putText(plotted, "VUNG ROI", (xmin + 10, ymin + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113), 2, cv2.LINE_AA)

            en_counts = _count_classes(res)
            vi_counts = _to_vi_counts(en_counts)
            total = int(sum(en_counts.values()))

            ok, buf = cv2.imencode(".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            await ws.send_text(
                json.dumps(
                    {
                        "image_b64": b64,
                        "counts_vi": vi_counts,
                        "counts_en": en_counts,
                        "total": total,
                        "ts": time.time(),
                        "seq": seq,
                    }
                )
            )
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


# ---------------- History ----------------
@app.post("/history")
async def add_history(
    counts_json: str = Form(...),
    lat: float | None = Form(None),
    lng: float | None = Form(None),
    note: str | None = Form(None),
    image: UploadFile | None = File(None),
) -> dict[str, Any]:
    rec_id = uuid.uuid4().hex
    image_name = None
    if image is not None:
        raw = await image.read()
        img = _decode_image(raw)
        image_name = _save_image_jpg(img, name_prefix="hist")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO history (id, created_at, lat, lng, note, counts_json, image_file) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (rec_id, time.time(), lat, lng, note, counts_json, image_name),
    )
    conn.commit()
    conn.close()

    return {
        "id": rec_id,
        "image_url": f"/files/{image_name}" if image_name else None,
    }


@app.get("/history")
def list_history(limit: int = 200) -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, created_at, lat, lng, note, counts_json, image_file FROM history ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            counts = json.loads(r[5]) if r[5] else {}
        except Exception:
            counts = {}
        out.append(
            {
                "id": r[0],
                "created_at": r[1],
                "lat": r[2],
                "lng": r[3],
                "note": r[4],
                "counts": counts,
                "image_url": f"/files/{r[6]}" if r[6] else None,
            }
        )
    return out


@app.delete("/history/{rec_id}")
def delete_history(rec_id: str) -> dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT image_file FROM history WHERE id = ?", (rec_id,)
    ).fetchone()
    conn.execute("DELETE FROM history WHERE id = ?", (rec_id,))
    conn.commit()
    conn.close()
    if row and row[0]:
        try:
            (STORAGE_DIR / row[0]).unlink(missing_ok=True)
        except Exception:
            pass
    return {"deleted": rec_id}


@app.get("/files/{name}")
def serve_file(name: str) -> FileResponse:
    storage_root = STORAGE_DIR.resolve()
    p = (STORAGE_DIR / name).resolve()
    try:
        p.relative_to(storage_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Khong tim thay file")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Khong tim thay file")
    return FileResponse(str(p))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
