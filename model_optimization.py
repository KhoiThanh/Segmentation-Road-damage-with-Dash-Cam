"""
YOLOv11m-seg → TFLite Mobile Export Script
===========================================
Exports trained model to optimized TFLite formats for on-device inference.

Variants exported:
  1. FP16  @ 416×416  — Recommended for real-time (GPU delegate)
  2. INT8  @ 416×416  — Smallest, fastest (CPU / NNAPI)
  3. FP16  @ 640×640  — High quality mode (GPU delegate only)

Usage:
  python model_optimization.py

Requirements:
  pip install ultralytics
"""

import os
import sys
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

# Path to your trained YOLOv11m-seg weights
MODEL_PATH = Path(
    "runs/segment/Road_Patches_Cracks_Segmentation_Datasets_yolo11l/weights/best.pt"
)

# Dataset YAML — required for INT8 calibration
DATA_YAML = Path("Road_Patches_Cracks_Segmentation_Datasets/data.yaml")

# Output directory (Flutter assets)
OUTPUT_DIR = Path("mobile_app/flutter_app/assets/models")

# Export configurations: (name, imgsz, half, int8)
EXPORT_CONFIGS = [
    ("best_float16",     416, True,  False),  # FP16 @ 416 — real-time
    ("best_int8",        416, False, True),    # INT8 @ 416 — smallest
    ("best_fp16_640",    640, True,  False),   # FP16 @ 640 — high quality
]


def main():
    # ─── Validate paths ──────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        logger.info("Please update MODEL_PATH in this script.")
        sys.exit(1)

    if not DATA_YAML.exists():
        logger.warning(f"Dataset YAML not found: {DATA_YAML}")
        logger.warning("INT8 export will be skipped (needs calibration data).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Load model ──────────────────────────────────────────────────────
    from ultralytics import YOLO

    logger.info(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # Print model info
    total_params = sum(p.numel() for p in model.model.parameters())
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"  Task:       {model.task}")
    logger.info(f"  Parameters: {total_params:,}")
    logger.info(f"  Size:       {model_size_mb:.1f} MB (.pt)")
    logger.info(f"  Classes:    {model.names}")

    # ─── Export each variant ─────────────────────────────────────────────
    results = []

    for name, imgsz, half, int8 in EXPORT_CONFIGS:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Exporting: {name}  (imgsz={imgsz}, half={half}, int8={int8})")
        logger.info("=" * 60)

        # Skip INT8 if no dataset YAML
        if int8 and not DATA_YAML.exists():
            logger.warning(f"  ⚠  Skipped — DATA_YAML not found for calibration")
            continue

        try:
            export_kwargs = dict(
                format="tflite",
                imgsz=imgsz,
                half=half,
                int8=int8,
            )
            # INT8 requires calibration data
            if int8:
                export_kwargs["data"] = str(DATA_YAML)

            export_path = model.export(**export_kwargs)

            # Copy to output directory with clean name
            dst = OUTPUT_DIR / f"{name}.tflite"
            shutil.copy2(export_path, dst)

            size_mb = dst.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ Saved: {dst}  ({size_mb:.1f} MB)")
            results.append((name, imgsz, size_mb))

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")

    # ─── Summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Export Summary")
    logger.info("=" * 60)
    logger.info(f"  Original model: {model_size_mb:.1f} MB (.pt)")
    for name, imgsz, size_mb in results:
        reduction = (1 - size_mb / model_size_mb) * 100
        logger.info(f"  {name}.tflite: {size_mb:.1f} MB  (imgsz={imgsz}, -{reduction:.0f}%)")

    logger.info("")
    logger.info(f"Models saved to: {OUTPUT_DIR.resolve()}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Copy TFLite files to Flutter assets/models/ (already done)")
    logger.info("  2. Run: cd mobile_app/flutter_app && flutter pub get")
    logger.info("  3. Run: flutter run  (on a physical Android device)")


if __name__ == "__main__":
    main()
