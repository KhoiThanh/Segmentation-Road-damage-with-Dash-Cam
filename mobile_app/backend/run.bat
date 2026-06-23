@echo off
REM Chay backend FastAPI cho ung dung mobile RDD
REM Su dung: doi vao thu muc backend roi go: run.bat

set RDD_MODEL_PATH=D:\Document\Code\DACN\RDDYOLOseg\runs\segment\Road_Patches_Cracks_Segmentation_Datasets_yolo11l\weights\best.pt
set PORT=8000

python -m uvicorn main:app --host 0.0.0.0 --port %PORT%
