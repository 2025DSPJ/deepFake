from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from PIL import Image as pil_image
import cv2
import tempfile
import os
import base64
import numpy as np
from network.models import model_selection  # xception 모델 불러오는 함수
from dataset.transform import xception_default_data_transforms
import gdown
from flask_cors import CORS
import requests
import uuid
import time
import matplotlib
matplotlib.use("Agg")  # 서버(무헤드) 환경에서 플롯 저장
import matplotlib.pyplot as plt

try:
    import dlib
    face_detector = dlib.get_frontal_face_detector()
    dlib_available = True
    print("[INFO] dlib successfully loaded. Dlib detector is available.")
except ImportError:
    face_detector = None
    dlib_available = False
    print("[WARN] dlib library not found. Dlib-based face detector will be unavailable.")


# ===== 설정 =====
SPRING_SERVER_URL = 'http://localhost:8080/progress'
MODEL_PATH = './model/xception.pth'
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "./outputs")  # 그래프 저장 루트
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 속도 목표 기본값(환경변수로 조정 가능)
TARGET_FPS_DEFAULT = float(os.environ.get("TARGET_FPS", "1.0"))         # 목표 처리량 (fps)
MAX_LAT_MS_DEFAULT = float(os.environ.get("MAX_LATENCY_MS", "2000"))    # 프레임당 최대 지연(ms)

# ===== 유틸: 진행률 보고 =====
def send_progress_to_spring(task_id, percent):
    try:
        payload = {'taskId': task_id, 'progress': percent}
        headers = {'Content-Type': 'application/json'}
        requests.post(SPRING_SERVER_URL, json=payload, headers=headers, timeout=1)
    except Exception as e:
        print(f"[WARN] 진행률 전송 실패: {e}")

# ===== 유틸: 모델 파일 확보 =====
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("모델이 없어서 Google Drive에서 다운로드")
        os.makedirs('./model', exist_ok=True)
        gdown.download(id='1j8AesqDjbSG0RfqaYaHdfGcVVkpdIPKJ', output=MODEL_PATH, quiet=False)

# ===== 유틸: 그림 저장 =====
def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_heatmap(per_frame_conf, out_path):
    if not per_frame_conf:
        return None
    arr = np.array(per_frame_conf, dtype=np.float32)[None, :]
    fig = plt.figure(figsize=(10, 2.0))
    ax = plt.subplot(111)
    im = ax.imshow(arr, aspect='auto', vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fake confidence (0–1)")
    ax.set_yticks([])
    ax.set_xlabel("Frame index")
    ax.set_title("Per-frame Fake Confidence Heatmap")
    fig.tight_layout()
    _save_fig(fig, out_path)
    return out_path

# ===== Flask 앱/모델 초기화 =====
ensure_model()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# face_detector = dlib.get_frontal_face_detector()

# OpenCV DNN face detector (백업/정밀 기본)
try:
    DNN_PROTO = "deploy.prototxt"
    DNN_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    face_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL) if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL) else None
except Exception as e:
    print(f"[WARN] DNN detector 초기화 실패: {e}")
    face_net = None

# 모델 불러오기
model = model_selection(modelname='xception', num_out_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
softmax = nn.Softmax(dim=1)

# ===== 공통 전처리 =====
def to_tensor_bgr(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    transform = xception_default_data_transforms['test']
    return transform(pil_image.fromarray(face_rgb)).unsqueeze(0)

# --- 조명 조건 판단(밝기+대비) & HSV 기반 보정(저강도) ---
def need_illum(gray):
    m, s = float(gray.mean()), float(gray.std())
    return (m < 70.0) or (s < 25.0)

def illum_hsv(face_bgr):
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
    v = clahe.apply(v)
    mid = max(np.median(v)/255.0, 1e-3)
    gamma = float(np.clip(np.log(0.5)/np.log(mid), 0.8, 1.4))
    lut = (np.linspace(0,1,256)**(1.0/gamma) * 255).astype(np.uint8)
    v = cv2.LUT(v, lut)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# --- 품질/조명 메트릭 ---
def var_laplace(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def luma_mean(gray):
    return float(gray.mean())

# ===== 기본 모드 유틸 =====
def predict_single_tensor(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        probs = softmax(output)[0]
        conf_fake = float(probs[1].item())
        pred = int(torch.argmax(probs).item())  # 0: REAL, 1: FAKE
    return pred, conf_fake

def predict(face_bgr):
    return predict_single_tensor(to_tensor_bgr(face_bgr))

# ===== 정밀 모드: 약한 기하/노출 브라케팅 TTA(가중 평균) =====
# 감마
def aug_gamma(img, gamma=1.05):
    inv = 1.0 / max(gamma, 1e-6)
    lut = (np.linspace(0,1,256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img, lut)

# 색상톤
def aug_hsv_jitter(img, dh=3, s_scale=1.03, v_scale=1.03):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = ((h.astype(np.int16) + dh) % 180).astype(np.uint8)
    s = np.clip(s.astype(np.float32) * s_scale, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.float32) * v_scale, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

# 가우시안 블러
def aug_gaussian_blur(img, sigma=0.6):
    k = max(3, int(2*round(3*sigma)+1))
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

# 언샤프 마스크(USM)
def aug_unsharp(img, amount=1.2, sigma=1.0):
    blur = cv2.GaussianBlur(img, (0,0), sigma)
    out = cv2.addWeighted(img, amount, blur, -(amount-1.0), 0)
    return out

def predict_tta_weighted(face_bgr, use_cond_illum=False):
    proc = face_bgr.copy()
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    _, conf_raw = predict_single_tensor(to_tensor_bgr(proc)) # RAW 단일 추론 (TTA 없이)
    if use_cond_illum and need_illum(gray):
        proc = illum_hsv(proc)
    # --- 게이팅: 40%-90% 확률에서만 TTA 실행 ---
    do_tta = True
    if not (0.4 <= conf_raw <= 0.9 or abs(conf_raw-0.6) <= 0.03):
        do_tta = False
    if not do_tta:
        return (1 if conf_raw >= 0.5 else 0), float(conf_raw)
    augs = [
        ("orig", lambda x: x, 0.5),
        ("hsv-", lambda x: aug_hsv_jitter(x, -3, 0.97, 0.97),    0.1),    
        ("gblur", lambda x: aug_gaussian_blur(x, 0.5), 0.13),
        ("usm",   lambda x: aug_unsharp(x, amount=1.2, sigma=1.0), 0.08),
        ("gamma", lambda x: aug_gamma(x, gamma=1.05),               0.2), 
    ]  

    # 가중치 정규화
    w_sum = sum(w for _,_,w in augs)
    augs = [(n,f,w/w_sum) for (n,f,w) in augs]
    vals, ws = [], []
    with torch.no_grad():
        for _, aug, w in augs:
            img = aug(proc.copy())
            _, conf_i = predict_single_tensor(to_tensor_bgr(img))
            vals.append(float(conf_i))
            ws.append(float(w))
    confs   = np.array(vals, dtype=np.float32)
    base_ws = np.array(ws,   dtype=np.float32)

    # 델타 기반 재가중
    delta = confs - float(conf_raw)
    k_push = 6.0
    w_adj = base_ws * np.exp(-k_push * np.maximum(0.0, delta))
    w_sum_adj = float(w_adj.sum())
    if w_sum_adj < 1e-9:
        w_adj = base_ws.copy()
        w_sum_adj = float(w_adj.sum())
    w_final = w_adj / w_sum_adj

    # 가중평균
    conf_tta = float((w_final * confs).sum()) 
    std_conf = float(np.std(confs)) # 분산 기반 보정
    if std_conf < 0.01: # std가 아주 작으면 TTA 무의미 → raw 사용
        conf_tta = conf_raw
    if std_conf > 0.05:  # std가 크면 과신 수축
        shrink = min(0.25, 5.0 * (std_conf - 0.05))
        conf_tta = 0.5 + (conf_tta - 0.5) * (1.0 - shrink)
    
    # 드리프트 캡
    max_drift = 0.02 + 0.30 * std_conf
    drift = conf_tta - conf_raw
    if abs(drift) > max_drift:
        conf_tta = conf_raw + np.sign(drift) * max_drift
    pred = 1 if conf_tta >= 0.5 else 0
    return pred, float(conf_tta)

def infer_prob(face_bgr, mode, use_tta, use_illum):
    if (mode == 'precision') and use_tta:
        _, p = predict_tta_weighted(face_bgr, use_cond_illum=bool(use_illum))
    else:
        _, p = predict(face_bgr)
    return float(p)

# ====== 안정성 지표 계산 ======
def tta_consistency_std(face_bgr, mode, use_tta, use_illum):
    augs = [
        lambda x: x,
        lambda x: cv2.flip(x, 1),
        lambda x: aug_hsv_jitter(x, -3, 0.97, 0.97),
        lambda x: aug_gaussian_blur(x, 0.5),
        lambda x: aug_unsharp(x, amount=1.2, sigma=1.0),
        lambda x: aug_gamma(x, gamma=1.05),
        lambda x: cv2.convertScaleAbs(x, alpha=1.05, beta=0),
        lambda x: cv2.convertScaleAbs(x, alpha=0.95, beta=0),
    ]
    ps = [infer_prob(aug(face_bgr), mode, use_tta, use_illum) for aug in augs]
    ps = np.array(ps, dtype=np.float32)
    return float(ps.std()), float(ps.mean())

def temporal_delta_stats(per_frame_conf):
    if len(per_frame_conf) < 2:
        return None, None
    arr = np.array(per_frame_conf, dtype=np.float32)
    diffs = np.abs(np.diff(arr))
    return float(diffs.mean()), float(diffs.std())

# ===== 얼굴 검출기: 다중 박스 반환 =====
def detect_face_bboxes(frame_bgr, detector='auto', dnn_conf=0.6, max_boxes=5):
    h, w = frame_bgr.shape[:2]
    boxes = []
    if detector in ('auto','dlib') and dlib_available and face_detector is not None:
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)
            for f in faces:
                x1,y1,x2,y2 = max(f.left(),0), max(f.top(),0), min(f.right(),w), min(f.bottom(),h)
                if (x2-x1)>0 and (y2-y1)>0:
                    boxes.append((x1,y1,x2,y2, 0.6))  # dummy conf
        except Exception as e:
            print(f"[WARN] dlib detect error: {e}")
    if detector in ('auto','dnn') and face_net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300,300)), 1.0, (300,300), (104,177,123))
        face_net.setInput(blob)
        detections = face_net.forward()
        for i in range(detections.shape[2]):
            conf = float(detections[0,0,i,2])
            if conf > dnn_conf:
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                x1,y1,x2,y2 = box.astype(int)
                x1,y1 = max(x1,0), max(y1,0)
                x2,y2 = min(x2,w), min(y2,h)
                if (x2-x1)>0 and (y2-y1)>0:
                    boxes.append((x1,y1,x2,y2, conf))
    boxes = list({(x1,y1,x2,y2):conf for x1,y1,x2,y2,conf in boxes}.items())
    boxes = [(b[0][0],b[0][1],b[0][2],b[0][3],b[1]) for b in boxes]
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes[:max_boxes]

def encode_jpg_base64(image_bgr):
    ok, buf = cv2.imencode('.jpg', image_bgr)
    if not ok:
        return None
    return base64.b64encode(buf).decode('utf-8')

# ===== robust 검출 =====
def robust_detect(frame, *, detector="dnn", dnn_conf=0.30, resize_long=720, max_boxes=5):
    H, W = frame.shape[:2]
    # 1차 DNN
    try:
        bboxes = detect_face_bboxes(frame, detector="dnn", dnn_conf=dnn_conf, max_boxes=max_boxes)
    except Exception as e:
        print("[ERR] dnn first pass:", repr(e), flush=True); 
        bboxes = []

    # 리사이즈 후 재시도
    if not bboxes:
        long_side = resize_long or 720
        scale = float(long_side) / float(max(H, W))
        fr = cv2.resize(frame, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        try:
            b2 = detect_face_bboxes(fr, detector="dnn", dnn_conf=0.25, max_boxes=max_boxes)
        except Exception as e:
            print("[ERR] dnn resized pass:", repr(e), flush=True); 
            b2 = []
        if b2:
            inv = (1.0/scale) if scale>0 else 1.0
            bboxes = [(int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv), conf) for (x1,y1,x2,y2,conf) in b2]
    
    # 폴백
    if not bboxes and detector != "dlib":
        try:
            fb = detect_face_bboxes(frame, detector="dlib", max_boxes=max_boxes) or []
            if fb:
                print("[DBG] fallback dlib hit", flush=True)
            bboxes = fb
        except Exception as e:
            print("[ERR] dlib fallback:", repr(e), flush=True); 
    return bboxes

# ===== API =====
@app.route('/predict', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    start_ts = time.time()
    per_frame_conf = []         # (스무딩된) 프레임별 확률 기록 (시각화용)
    raw_conf_for_vote = []      # 집계용: (p_i, q_i, l_i)

    video_file = request.files['file']
    task_id = request.form.get('taskId') or str(uuid.uuid4())
    print(f"[INFO] taskId={task_id}")

    mode = request.form.get('mode', 'default')  # "default" | "precision"
    use_tta = request.form.get('use_tta')
    use_illum = request.form.get('use_illum')
    detector = request.form.get('detector') or ('dnn' if mode=='precision' else 'auto')  # 정밀 기본 DNN
    smooth_window = int(request.form.get('smooth_window', 0) or 0)
    min_face = int(request.form.get('min_face', 96 if mode=='precision' else 64) or (96 if mode=='precision' else 64))
    sample_count = int(request.form.get('sample_count', 20 if mode=='precision' else 10) or (20 if mode=='precision' else 10))

    # 속도 목표 파라미터(요청이 우선, 없으면 환경변수 기본)
    target_fps = float(request.form.get('target_fps', TARGET_FPS_DEFAULT))
    max_latency_ms = float(request.form.get('max_latency_ms', MAX_LAT_MS_DEFAULT))
    # if mode == 'default':
    #     if 'target_fps' not in request.form:
    #         target_fps = 0.35     # 기본모드 목표 처리량
    #     if 'max_latency_ms' not in request.form:
    #         max_latency_ms = 3000.0  # 기본모드 지연 목표
    # elif mode == 'precision':
    #     if 'target_fps' not in request.form:
    #         target_fps = 0.20
    #     if 'max_latency_ms' not in request.form:
    #         max_latency_ms = 5000.0
    target_fps = 0.27     # 기본모드 목표 처리량
    max_latency_ms = 4000.0  # 기본모드 지연 목표

    # 프리셋
    if mode == 'precision':
        use_tta = True if use_tta is None else (use_tta.lower() == 'true')
        use_illum = True if use_illum is None else (use_illum.lower() == 'true')
        smooth_window = smooth_window or 0   # EMA로 대체
        sample_count = sample_count or 20
        if (detector == 'dnn') and (face_net is None):
            if dlib_available and face_detector is not None:
                print("[WARN] DNN face_net unavailable. Falling back to dlib for precision mode.")
                detector = 'dlib'
            else:
                print("[ERROR] No face detector available for precision mode (DNN missing, dlib unavailable).")
    else:
        use_tta = False if use_tta is None else (use_tta.lower() == 'true')
        use_illum = False if use_illum is None else (use_illum.lower() == 'true')
        smooth_window = smooth_window or 0
        sample_count = sample_count or 10

    # 동영상 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_path = tmp.name
        try:
            video_file.stream.seek(0)
        except Exception:
            pass
        video_file.save(tmp)

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames == 0:
        cap.release()
        os.remove(video_path)
        return jsonify({'error': 'Invalid video or zero frames'}), 400

    # 균등 샘플링
    step = max(1, num_frames // max(1, sample_count))
    target_indices = set([min(i*step, num_frames-1) for i in range(max(1, sample_count))])

    results = []
    max_confidence = -1.0
    max_conf_frame = None
    processed_frames = 0
    expected = len(target_indices)

    ema = None  # EMA 스무딩
    frame_idx = 0
    scene_lumas = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx in target_indices:
            try:
                if frame.dtype != 'uint8':
                    frame = frame.astype('uint8')

                if mode == 'precision':
                    # ---- 정밀 모드 ----
                    bboxes = robust_detect(frame, detector=detector)

                    if not bboxes and os.getenv("BYPASS_DETECT", "0") == "1":
                        H, W = frame.shape[:2]
                        cx, cy = W // 2, H // 2
                        s = min(H, W) // 3
                        bboxes = [(cx - s, cy - s, cx + s, cy + s, 1.0)]

                    if bboxes:
                        bboxes_sorted_area = sorted(bboxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)
                        primary = bboxes_sorted_area[0]
                        bboxes_sorted_conf = sorted(bboxes, key=lambda b:b[4], reverse=True)
                        secondary = bboxes_sorted_conf[0]
                        chosen = [primary]
                        if secondary != primary and len(bboxes_sorted_area)>1:
                            chosen.append(secondary)
                        frame_best_conf = 0.0

                        for (x1,y1,x2,y2, conf_det) in chosen[:2]:
                            face_img = frame[y1:y2, x1:x2]
                            if face_img.size == 0 or min(face_img.shape[:2]) < min_face:
                                continue
                            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                            q = var_laplace(gray_face)
                            l = luma_mean(gray_face)

                            pred_i, conf_i = (
                                predict_tta_weighted(face_img, use_cond_illum=use_illum)
                                if use_tta else
                                predict(face_img)
                            )

                            # EMA 스무딩 (프레임 대표 확률에 적용)
                            if ema is None:
                                ema = conf_i
                            else:
                                ema = 0.5*conf_i + 0.5*ema
                            conf_s = float(ema)

                            raw_conf_for_vote.append( (conf_i, q, l) )
                            per_frame_conf.append(conf_s)
                            results.append({'pred': 1 if conf_s>=0.5 else 0, 'confidence': conf_s})

                            if conf_i > frame_best_conf:
                                frame_best_conf = conf_i
                                if conf_i > max_confidence:
                                    max_confidence = conf_i
                                    max_conf_frame = face_img.copy()

                            scene_lumas.append(l)

                else:
                    # ---- 기본 모드 ----
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detector(gray)
                    if faces:
                        x1 = max(faces[0].left(), 0)
                        y1 = max(faces[0].top(), 0)
                        x2 = min(faces[0].right(), frame.shape[1])
                        y2 = min(faces[0].bottom(), frame.shape[0])
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size > 0 and min(face_img.shape[:2]) >= 64:
                            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                            q = var_laplace(gray_face)
                            l = luma_mean(gray_face)
                            pred, conf = predict(face_img)

                            if ema is None:
                                ema = conf
                            else:
                                ema = 0.5*conf + 0.5*ema
                            conf_s = float(ema)

                            raw_conf_for_vote.append( (conf, q, l) )
                            per_frame_conf.append(conf_s)
                            results.append({'pred': 1 if conf_s>=0.5 else 0, 'confidence': conf_s})

                            if conf > max_confidence:
                                max_confidence = conf
                                max_conf_frame = face_img.copy()

                            scene_lumas.append(l)

            except Exception as e:
                print(f"Error in detection/prediction: {e}")

            processed_frames += 1
            progress_percent = int(100 * processed_frames / max(1, expected))
            send_progress_to_spring(task_id, progress_percent)

        frame_idx += 1

    cap.release()
    os.remove(video_path)

    if not results:
        return jsonify({
            'result': 'no face detected',
            'mode': mode,
            'use_tta': use_tta,
            'use_illum': use_illum,
            'detector': detector,
            'smooth_window': smooth_window,
            'min_face': min_face,
            'sample_count': sample_count,
            'taskId': task_id
        }), 200

    # ===== 집계: 가중 투표 =====
    # weights, scores = [], []
    # for (p, q, l) in raw_conf_for_vote:
    #     w_conf = 0.5 + abs(p - 0.5)             # 0.5~1.0
    #     w_quality = min(1.0, q / 100.0)         # 대략 0~1
    #     w_light = 1.0
    #     if l < 40 or l > 210:                   # 극저조/과다노출
    #         w_light = 0.85
    #     w = w_conf * w_quality * w_light
    #     weights.append(w)
    #     scores.append(p)
    # S = float(np.dot(weights, scores) / max(np.sum(weights), 1e-6))
     # ==== CHANGED: 강건 집계 + 고신뢰 프레임 증거 도입 ======================
     # raw_conf_for_vote 항목이 (p, q, l) 또는 (p, q, l, astd, area, dconf)일 수 있으니
     # 앞 3개만 안전하게 꺼냄
    weights, scores, conf_list = [], [], [r['confidence'] for r in results]
 
    # 1) 기본 가중 (품질/조도 + 확신도)
    for tup in raw_conf_for_vote:
        p, q, l = tup[0], tup[1], tup[2]
        w_conf   = 0.5 + abs(p - 0.5)                  # 0.5~1.0
        w_quality= min(1.0, q / 110.0)                 # 품질 상향 기준
        w_light  = 0.9 if (l < 40 or l > 210) else 1.0 # 극단 조명 패널티
        w = w_conf * w_quality * w_light
        weights.append(w)
        scores.append(p)
 
    # 2) 가중 평균
    S_mean = float(np.dot(weights, scores) / max(np.sum(weights), 1e-6)) if weights else 0.0
 
    # 3) 절단평균(상/하위 10% 제거)로 극단값 영향 억제
    conf_sorted = sorted(conf_list)
    if len(conf_sorted) >= 10:
        k = max(1, int(0.10 * len(conf_sorted)))
        trimmed = conf_sorted[k:len(conf_sorted)-k]
        S_trim  = float(np.mean(trimmed)) if trimmed else S_mean
    else:
        S_trim = S_mean
 
    # 4) 고신뢰 프레임 증거(두 가지 중 하나 충족 시 FAKE로 강하게 지지)
    frac_high = (sum(c >= 0.85 for c in conf_list) / max(1, len(conf_list))) if conf_list else 0.0
    # 짧은 강한 구간(연속 3프레임 0.75↑) 체크
    streak3 = any(conf_list[i] >= 0.8 and conf_list[i+1] >= 0.8 and conf_list[i+2] >= 0.8
                  for i in range(0, max(0, len(conf_list)-2)))
     # =======================================================================
  
    tau = 0.50
    if len(scene_lumas) > 0 and (np.mean(scene_lumas) < 70 or np.mean(scene_lumas) > 210):
        tau = 0.48  # 저조/과다노출에서 약간 완화
    # final_label = 'FAKE' if S >= tau else 'REAL'
     # ==== CHANGED: 이중 임계 + 고신뢰 증거 + 보류(UNCERTAIN) =================
    if mode == 'precision':
        tau_low, tau_high = 0.50, 0.58   # 정밀: 오탐 억제는 보조규칙에 맡기고 τ는 과도하게 올리지 않음
        need_frac_high    = 0.25         # conf≥0.8 프레임이 25% 이상이면 FAKE 지지
    else:
        tau_low, tau_high = 0.50, 0.62   # 기본: 속도모드라 평균이 높아도 보수적으로 한 번 더 확인
        need_frac_high    = 0.35         # 기본은 더 많은 고신뢰 프레임 요구
 
    S = S_trim  # 최종 판단은 절단평균 기반
    print(S)
 
    final_label = 'UNCERTAIN'
    if (S >= tau_high and (frac_high >= need_frac_high or streak3)):
        final_label = 'FAKE'
    else:
        final_label = 'REAL'


    # 대표 프레임(base64)
    if max_conf_frame is not None:
        ok, buffer = cv2.imencode('.jpg', max_conf_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8') if ok else None
    else:
        img_base64 = None

    #축소 함수
    def shrink_value_scalar(x, alpha=2.5):
        x = float(np.clip(x, 0.0, 1.0))
        return x ** alpha

    def shrink_value_vec(arr, alpha=2.5):
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        return np.power(arr, alpha, dtype=np.float32)

    # ===== 통계/지표 =====
    conf_arr = np.array(per_frame_conf, dtype=np.float32) if per_frame_conf else np.zeros((0,), dtype=np.float32)
    # --- 절단 전처리(상위 10% 제거) ---
    if conf_arr.size >= 10:
        k_trim = max(1, int(0.10 * conf_arr.size))
        conf_sorted = np.sort(conf_arr)
        conf_trimmed = conf_sorted[:conf_arr.size-k_trim] if (2*k_trim) < conf_sorted.size else conf_sorted
    else:
        k_trim = 0
        conf_trimmed = conf_arr  # 샘플이 적으면 원본 유지

    avg_confidence = float(conf_trimmed.mean()) if conf_trimmed.size else None
    max_confidence = float(conf_trimmed.max()) if conf_trimmed.size else None
    median_confidence = float(np.median(conf_arr)) if conf_arr.size else None
    variance_confidence = float(conf_arr.var()) if conf_arr.size else None
    frame_vote_ratio = float(sum(1 for r in results if r['pred'] == 1)) / float(len(results))
    processing_time_sec = time.time() - start_ts
    fps_processed = (len(per_frame_conf)/processing_time_sec) if processing_time_sec > 0 else None
    ms_per_sample = ((processing_time_sec / len(per_frame_conf)) * 1000.0) if len(per_frame_conf) > 0 else None
    print(f"avg_confidence: {avg_confidence}, max_confidence: {max_confidence}, final_label: {final_label}")
    
    # 축소 통계 
    ALPHA =1.2
    conf_trimmed_shr = shrink_value_vec(conf_trimmed, alpha=ALPHA)

    avg_confidence = float(conf_trimmed_shr.mean()) if conf_trimmed_shr.size else 0.0
    max_confidence = float(conf_trimmed_shr.max())  if conf_trimmed_shr.size else None

    # 히트맵 저장 (taskId별 폴더)
    task_dir = os.path.join(OUTPUT_ROOT, task_id)
    heatmap_path = os.path.join(task_dir, f"heatmap_{task_id}.png")
    try:
        per_frame_conf_shr = [float(shrink_value_scalar(x, alpha=ALPHA)) for x in per_frame_conf]  # 0~1 범위 유지
        save_heatmap(per_frame_conf_shr, heatmap_path)
    except Exception as e:
        print(f"[WARN] heatmap save failed: {e}")
        heatmap_path = None

    # ===== 안정성 지표 (대표 프레임 1장 기준 + 시계열 기준) =====
    tta_std = tta_mean = None
    temporal_mean = temporal_std = None
    if max_conf_frame is not None:
        try:
            tta_std, tta_mean = tta_consistency_std(max_conf_frame, mode, use_tta, use_illum)
        except Exception as e:
            print(f"[WARN] tta_consistency failed: {e}")
        try:
            temporal_mean, temporal_std = temporal_delta_stats(per_frame_conf)
        except Exception as e:
            print(f"[WARN] temporal stats failed: {e}")

    # 속도 합격 여부
    speed_ok = (
        (fps_processed or 0.0) >= float(target_fps)
        and (ms_per_sample or 1e12) <= float(max_latency_ms)
    )

    return jsonify({
        'taskId': task_id,
        'result': final_label,        
        'most_suspect_image': img_base64,  # base64 encoded image

        'score_weighted': round(S, 4),
        'threshold_tau': tau_high,
        'frame_vote_ratio': round(frame_vote_ratio, 4),

        'average_fake_confidence': round(avg_confidence, 4),
        'median_confidence': round(median_confidence, 4) if median_confidence is not None else None,
        'variance_confidence': round(variance_confidence, 6) if variance_confidence is not None else None,
        'max_confidence': round(max_confidence, 4) if max_confidence >= 0 else None,
        
        'frames_processed': len(per_frame_conf),
        'processing_time_sec': round(processing_time_sec, 3),
        
        # 실행 환경
        'mode': mode,
        'use_tta': use_tta,
        'use_illum': use_illum,
        'detector': detector,
        'smooth_window': smooth_window,
        'min_face': min_face,
        'sample_count': sample_count,
        # 히트맵
        'timeseries': {
            'per_frame_conf': [float(x) for x in per_frame_conf],
            'vmin': 0.0,
            'vmax': 1.0
        },
        # 안정성 원시 값
        'stability_evidence': {
            'tta_std': round(tta_std, 6) if tta_std is not None else None,
            'tta_mean': round(tta_mean, 6) if tta_mean is not None else None,
            'temporal_delta_mean': round(temporal_mean, 6) if temporal_mean is not None else None,
            'temporal_delta_std': round(temporal_std, 6) if temporal_std is not None else None
        },
       'speed': {
            'fps_processed': round(fps_processed, 3) if fps_processed is not None else None,
            'ms_per_sample': round(ms_per_sample, 1) if ms_per_sample is not None else None,
            'target_fps': float(target_fps),
            'max_latency_ms': float(max_latency_ms),
            'speed_ok': bool(speed_ok),
        }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
