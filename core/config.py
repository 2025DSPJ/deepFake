import os

SPRING_SERVER_URL = os.environ.get("SPRING_SERVER_URL", "http://localhost:8080/progress")
MODEL_PATH        = os.environ.get("MODEL_PATH", "./model/xception.pth")
OUTPUT_ROOT       = os.environ.get("OUTPUT_ROOT", "./outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 속도목표(요청 값 없을 때 기본)
TARGET_FPS_DEFAULT  = float(os.environ.get("TARGET_FPS", "0.27"))
MAX_LAT_MS_DEFAULT  = float(os.environ.get("MAX_LATENCY_MS", "2000"))

# 얼굴 DNN 파일(있으면 사용)
DNN_PROTO = "deploy.prototxt"
DNN_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
