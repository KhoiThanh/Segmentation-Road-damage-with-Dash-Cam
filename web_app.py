import gradio as gr
from ultralytics import YOLO
import cv2
import tempfile
import os
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F

# Khởi tạo mô hình YOLO
model = YOLO(r"D:\Document\Code\DACN\RDDYOLOseg\runs\segment\Road_Patches_Cracks_Segmentation_Datasets_yolo11l\weights\best.pt") 

# Cấu hình ngưỡng cố định
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.45

def _format_detection_summary(result) -> str:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return "Tổng detections: 0"

    try:
        cls_list = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    except Exception:
        try:
            cls_list = [int(x) for x in boxes.cls]
        except Exception:
            cls_list = []

    total = len(cls_list)
    if total == 0:
        return "Tổng detections: 0"

    names = getattr(result, "names", None) or getattr(model, "names", {})
    counts = Counter(cls_list)
    lines = [f"Tổng detections: {total}"]
    for cls_id, count in sorted(counts.items(), key=lambda kv: kv[0]):
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        lines.append(f"- {cls_name}: {count}")
    return "\n".join(lines)

def predict_image(image, roi_enabled, roi_top, roi_bottom, roi_left, roi_right):
    if image is None:
        return None, None, None, ""
    # Thực hiện dự đoán trên ảnh với threshold cố định 0.2
    conf = 0.2
    iou = 0.45
    results = model(image, conf=conf, iou=iou)
    result = results[0]
    
    height, width = image.shape[:2]
    
    # Áp dụng ROI nếu được kích hoạt và hợp lệ
    has_valid_roi = roi_enabled and (roi_top < roi_bottom) and (roi_left < roi_right)
    if has_valid_roi:
        ymin = int(height * roi_top / 100)
        ymax = int(height * roi_bottom / 100)
        xmin = int(width * roi_left / 100)
        xmax = int(width * roi_right / 100)
        
        keep_indices = []
        boxes = getattr(result, "boxes", None)
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
        result = result[keep_indices]
    
    # Lấy ảnh kết quả đã được vẽ bounding box và mask
    res_plotted = result.plot()
    
    # Vẽ khung vùng ROI lên ảnh kết quả
    if has_valid_roi:
        ymin = int(height * roi_top / 100)
        ymax = int(height * roi_bottom / 100)
        xmin = int(width * roi_left / 100)
        xmax = int(width * roi_right / 100)
        
        # Vẽ đường viền ROI màu xanh lá cây
        cv2.rectangle(res_plotted, (xmin, ymin), (xmax, ymax), (46, 204, 113), 2)
        # Thêm nhãn "VÙNG ROI" ở góc trên bên trái của vùng ROI
        cv2.putText(res_plotted, "VUNG ROI", (xmin + 10, ymin + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113), 2, cv2.LINE_AA)
                    
    summary = _format_detection_summary(result)
    
    # Tạo Eigen-CAM heatmap (Class-Agnostic, không phụ thuộc lớp mục tiêu)
    try:
        overlay, heatmap, cam_info = _generate_eigencam(
            image,
            colormap=cv2.COLORMAP_JET,
            alpha=0.5,
        )
        summary += "\n\n--- Eigen-CAM ---\n" + cam_info
    except Exception as e:
        overlay = image.copy()
        heatmap = None
        summary += f"\n\n--- Eigen-CAM ---\nLỗi: {str(e)}"
    
    return res_plotted, overlay, heatmap, summary

def predict_video(video_path, roi_enabled, roi_top, roi_bottom, roi_left, roi_right):
    # Sử dụng threshold cố định
    conf = 0.2
    iou = 0.45
    
    if video_path is None:
        return None, ""

    if isinstance(video_path, dict):
        video_path = video_path.get("name") or video_path.get("path")
    if not isinstance(video_path, str) or not os.path.exists(video_path):
        return None, "Không tìm thấy file video."
    
    # Đọc video
    cap = cv2.VideoCapture(video_path)
    
    # Lấy thông số video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Tạo file tạm để lưu video kết quả
    fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames = 0
    frames_with_detections = 0
    cumulative_counts = Counter()

    # Tính toán tọa độ ROI nếu bật
    has_valid_roi = roi_enabled and (roi_top < roi_bottom) and (roi_left < roi_right)
    if has_valid_roi:
        ymin = int(height * roi_top / 100)
        ymax = int(height * roi_bottom / 100)
        xmin = int(width * roi_left / 100)
        xmax = int(width * roi_right / 100)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
            
        # Dự đoán trên từng frame với threshold
        results = model(frame, conf=conf, iou=iou)
        result = results[0]
        
        if has_valid_roi:
            keep_indices = []
            boxes = getattr(result, "boxes", None)
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
            result = result[keep_indices]

        res_frame = result.plot()
        
        if has_valid_roi:
            # Vẽ đường viền ROI màu xanh lá cây
            cv2.rectangle(res_frame, (xmin, ymin), (xmax, ymax), (46, 204, 113), 2)
            # Thêm nhãn "VÙNG ROI" ở góc trên bên trái của vùng ROI
            cv2.putText(res_frame, "VUNG ROI", (xmin + 10, ymin + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113), 2, cv2.LINE_AA)

        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.cls is not None:
            try:
                cls_list = boxes.cls.detach().cpu().numpy().astype(int).tolist()
            except Exception:
                cls_list = []
            if cls_list:
                frames_with_detections += 1
                cumulative_counts.update(cls_list)
        
        out.write(res_frame)
        
    cap.release()
    out.release()
    
    names = getattr(model, "names", {})
    total = sum(cumulative_counts.values())
    lines = [
        f"Tổng detections (đếm theo từng frame): {total}",
        f"Số frames: {frames}",
        f"Frames có detections: {frames_with_detections}",
    ]
    for cls_id, count in sorted(cumulative_counts.items(), key=lambda kv: kv[0]):
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        lines.append(f"- {cls_name}: {count}")
    summary = "\n".join(lines)

    return output_path, summary

# ---------------------------------------------------------------------------
# Grad-CAM cho YOLO  
# ---------------------------------------------------------------------------

def _find_target_layer(yolo_model):
    """Tìm target layer phù hợp (C3k2/C2f ở thang đo P3 để bắt được cả vết nứt nhỏ và ổ gà) để làm Grad-CAM."""
    try:
        backbone = yolo_model.model.model  # nn.Sequential chứa các module
        num_layers = len(backbone)
        if num_layers == 24:  # YOLOv11
            return backbone[16]  # C3k2 ở P3 scale (độ phân giải cao 80x80)
        elif num_layers == 23:  # YOLOv8
            return backbone[15]  # C2f ở P3 scale
        else:
            return backbone[-2]  # Fallback lấy lớp trước head
    except Exception:
        return None




def _generate_eigencam(
    image: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Tính Eigen-CAM heatmap cho ảnh đầu vào bằng cách phân tích SVD trên activations.
    Eigen-CAM là phương pháp Class-Agnostic, hoạt động không cần lan truyền ngược gradient.
    
    Returns:
        overlay   – ảnh gốc + heatmap chồng lên (RGB)
        heatmap   – bản đồ nhiệt thuần (RGB)
        info_text – thông tin chi tiết
    """
    yolo_model = model  # YOLO instance toàn cục
    pt_model = yolo_model.model  # DetectionModel/SegmentationModel (xử lý skip connections đúng)
    
    target_layer = _find_target_layer(yolo_model)
    if target_layer is None:
        raise RuntimeError("Không tìm được target layer phù hợp để tính Eigen-CAM.")

    # ---------- Register forward hook ----------
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    fh = target_layer.register_forward_hook(forward_hook)

    try:
        # ---------- Chuẩn bị ảnh ----------
        h_orig, w_orig = image.shape[:2]

        with torch.inference_mode():
            # Resize về 640×640 (input chuẩn YOLO), giữ tensor
            img_resized = cv2.resize(image, (640, 640))
            tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            device = next(pt_model.parameters()).device
            tensor = tensor.to(device)

            # ---------- Forward pass ----------
            # Dùng DetectionModel.forward() để trích xuất đặc trưng
            pt_model.eval()
            _ = pt_model(tensor)

        if not activations:
            raise RuntimeError("Không thu được activation từ target layer.")

        # Lấy activations [1, C, H, W]
        act = activations[0]
        
        # Thay thế NaNs nếu có
        act = torch.nan_to_num(act, nan=0.0)

        # Chạy SVD trực tiếp bằng PyTorch
        C, H, W = act[0].shape
        reshaped = act[0].view(C, -1).t() # (H*W, C)
        centered = reshaped - reshaped.mean(dim=0, keepdim=True)

        # SVD: centered = U * S * Vh
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        # Chiếu dữ liệu lên vector riêng đầu tiên (Vh[0, :])
        projection = centered @ Vh[0, :]  # (H*W,)
        projection = projection.view(H, W)

        # Xử lý lật dấu (sign flip) để đảm bảo vùng kích hoạt chính mang giá trị dương
        if projection.sum() < 0:
            projection = -projection

        # Chuẩn hóa về [0, 1]
        proj_min = projection.min()
        proj_max = projection.max()
        if proj_max - proj_min > 1e-8:
            cam = (projection - proj_min) / (proj_max - proj_min)
        else:
            cam = torch.zeros_like(projection)

        # Chuyển về numpy, resize về kích thước gốc
        cam_np = cam.detach().cpu().numpy()  # (H, W)
        cam_resized = cv2.resize(cam_np, (w_orig, h_orig))  # (h_orig, w_orig)

        # Tạo heatmap màu (applyColorMap trả về BGR)
        heatmap_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        # Chuyển đổi sang RGB để đồng bộ với ảnh gốc của Gradio
        heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

        # Chồng lên ảnh gốc
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color_rgb, alpha, 0)

        # Thông tin
        layer_name = target_layer.__class__.__name__
        info_lines = [
            f"Target layer: {layer_name}",
            f"Feature map size: {act.shape[2]}×{act.shape[3]}",
            f"Channels: {act.shape[1]}",
            f"Phương pháp: Eigen-CAM (Class-Agnostic)",
            f"SVD Singular Values (Top 3): {', '.join([f'{s.item():.2f}' for s in S[:3]])}",
            f"Heatmap range: [{cam_np.min():.3f}, {cam_np.max():.3f}]",
        ]
        info_text = "\n".join(info_lines)

        return overlay, heatmap_color_rgb, info_text

    finally:
        fh.remove()
        activations.clear()


# Mapping colormap tên → hằng số OpenCV
COLORMAP_OPTIONS = {
    "JET (cầu vồng)": cv2.COLORMAP_JET,
    "HOT (nóng)": cv2.COLORMAP_HOT,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "TURBO": cv2.COLORMAP_TURBO,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
}


def eigencam_predict(image, colormap_name, alpha):
    """Hàm xử lý cho tab Eigen-CAM trên Gradio."""
    if image is None:
        return None, None, ""
    
    colormap = COLORMAP_OPTIONS.get(colormap_name, cv2.COLORMAP_JET)
    
    try:
        overlay, heatmap, info = _generate_eigencam(
            image,
            colormap=colormap,
            alpha=alpha,
        )
        return overlay, heatmap, info
    except Exception as e:
        return image, None, f"Lỗi: {str(e)}"


# CSS tùy chỉnh để cải thiện thẩm mỹ
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
h1 {
    font-weight: 800 !important;
    background: linear-gradient(90deg, #2563eb 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2em !important;
}
.roi-container {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 15px;
    background-color: #f8fafc;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="emerald",
    neutral_hue="slate"
)

# Xây dựng giao diện Gradio
with gr.Blocks(title="Ứng dụng Nhận diện Hư hỏng Đường bộ") as demo:
    gr.Markdown("# Ứng dụng Nhận diện Hư hỏng Đường bộ với YOLO")
    gr.Markdown(
        "<p style='text-align: center; font-size: 1.1em; color: #475569; margin-top: -10px;'>"
        "Hỗ trợ nhận dạng vết nứt (Crack) và ổ gà (Pothole) trên hình ảnh và video với cấu hình vùng lọc ROI thông minh."
        "</p>"
    )
    
    with gr.Accordion("🛠️ Cấu hình vùng quan tâm (ROI - Region of Interest)", open=False):
        gr.Markdown(
            "Thiết lập vùng ROI để chỉ nhận diện hư hỏng đường bộ trong phạm vi nhất định (ví dụ: loại bỏ phần trời, cảnh bên đường hoặc mui xe ô tô)."
        )
        with gr.Row():
            roi_enabled = gr.Checkbox(label="Kích hoạt lọc theo vùng ROI", value=False)
        with gr.Row():
            roi_top = gr.Slider(0, 100, value=0, step=1, label="Cắt mép trên (%)")
            roi_bottom = gr.Slider(0, 100, value=100, step=1, label="Cắt mép dưới (%)")
        with gr.Row():
            roi_left = gr.Slider(0, 100, value=0, step=1, label="Cắt mép trái (%)")
            roi_right = gr.Slider(0, 100, value=100, step=1, label="Cắt mép phải (%)")
    
    with gr.Tabs():
        # Tab cho Hình ảnh
        with gr.TabItem("Nhận diện Hình ảnh"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy", label="Tải ảnh lên")
                    image_button = gr.Button("Nhận diện Ảnh", variant="primary")
                with gr.Column():
                    image_output = gr.Image(type="numpy", label="Kết quả Phân vùng")
            with gr.Row():
                with gr.Column():
                    cam_overlay_out = gr.Image(type="numpy", label="🔥 Eigen-CAM Overlay")
                with gr.Column():
                    cam_heatmap_out = gr.Image(type="numpy", label="🔥 Heatmap thuần")
            image_counts = gr.Textbox(label="Số lượng hư hỏng phát hiện & Eigen-CAM", lines=10)
            
            image_button.click(
                predict_image, 
                inputs=[image_input, roi_enabled, roi_top, roi_bottom, roi_left, roi_right], 
                outputs=[image_output, cam_overlay_out, cam_heatmap_out, image_counts]
            )
            
        # Tab cho Video
        with gr.TabItem("Nhận diện Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Tải video lên")
                    video_button = gr.Button("Nhận diện Video", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Kết quả Video")
                    video_counts = gr.Textbox(label="Số lượng hư hỏng phát hiện", lines=10)
                    
            video_button.click(
                predict_video, 
                inputs=[video_input, roi_enabled, roi_top, roi_bottom, roi_left, roi_right], 
                outputs=[video_output, video_counts]
            )

        # Tab cho Eigen-CAM Heatmap
        with gr.TabItem("🔥 Eigen-CAM Heatmap"):
            gr.Markdown(
                "<p style='color: #475569; font-size: 0.95em;'>"
                "Trực quan hóa vùng ảnh mà mô hình <b>tập trung chú ý</b> bằng kỹ thuật <b>Eigen-CAM</b> "
                "(Phân tích thành phần chính PCA trên bản đồ đặc trưng). "
                "Eigen-CAM hoạt động độc lập với nhãn lớp (Class-Agnostic) và không sử dụng lan truyền ngược gradient, "
                "giúp hiển thị trực quan các đặc trưng kết cấu và biên dạng vật thể một cách cực kỳ ổn định và nhanh chóng."
                "</p>"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    cam_input = gr.Image(type="numpy", label="Tải ảnh lên")
                    with gr.Row():
                        cam_colormap = gr.Dropdown(
                            choices=list(COLORMAP_OPTIONS.keys()),
                            value="JET (cầu vồng)",
                            label="Bảng màu (Colormap)",
                        )
                    cam_alpha = gr.Slider(
                        0.1, 0.9, value=0.5, step=0.05,
                        label="Độ trong suốt heatmap (alpha)",
                    )
                    cam_button = gr.Button("🔥 Tạo Heatmap Eigen-CAM", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        cam_overlay = gr.Image(type="numpy", label="Ảnh gốc + Heatmap")
                        cam_heatmap = gr.Image(type="numpy", label="Heatmap thuần")
                    cam_info = gr.Textbox(label="Thông tin Eigen-CAM", lines=7)

            cam_button.click(
                eigencam_predict,
                inputs=[cam_input, cam_colormap, cam_alpha],
                outputs=[cam_overlay, cam_heatmap, cam_info],
            )


if __name__ == "__main__":
    demo.launch(theme=theme, css=custom_css)