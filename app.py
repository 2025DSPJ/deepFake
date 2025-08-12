from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image as pil_image
import cv2
import dlib
import tempfile
import os
import io
import base64
import numpy as np
from tqdm import tqdm
from network.models import model_selection  # xception 모델 불러오는 함수
from dataset.transform import xception_default_data_transforms
import gdown
from flask_cors import CORS
import requests
import uuid
from collections import deque

SPRING_SERVER_URL = 'http://localhost:8080/progress' 
model_path = './model/xception.pth'

def send_progress_to_spring(task_id, percent):
    try:
        payload = {
            'taskId': task_id,
            'progress': percent
        }
        headers = {
            'Content-Type': 'application/json'
        }
        requests.post(SPRING_SERVER_URL, json=payload, headers=headers, timeout=1)
    except Exception as e:
        print(f"[WARN] 진행률 전송 실패: {e}")

def ensure_model():
    # 모델이 없을 경우, Google Drive에서 다운로드
    if not os.path.exists(model_path):
        print("모델이 없어서 Google Drive에서 다운로드")
        os.makedirs('./model', exist_ok=True)
        gdown.download(id='1j8AesqDjbSG0RfqaYaHdfGcVVkpdIPKJ', output=model_path, quiet=False)

ensure_model()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
face_detector = dlib.get_frontal_face_detector()

# OpenCV DNN face detector (백업용)
try:
    DNN_PROTO = "deploy.prototxt"
    DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
    face_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL) if os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL) else None
except Exception as e:
    print(f"[WARN] DNN detector 초기화 실패: {e}")
    face_net = None

# 모델 불러오기
model = model_selection(modelname='xception', num_out_classes=2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
softmax = nn.Softmax(dim=1)


# ---------------------------
# 기본 모드용 유틸
# ---------------------------

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = xception_default_data_transforms['test']
    img_tensor = transform(pil_image.fromarray(image)).unsqueeze(0)
    return img_tensor

def predict(image):
    with torch.no_grad():
        input_tensor = preprocess_image(image)
        output = model(input_tensor)
        probs = nn.Softmax(dim=1)(output)
        confidence = probs[0][1].item()  # FAKE 확률
        pred = torch.argmax(probs, dim=1).item()
    return pred, confidence

# ---------------------------
# 정밀 모드용 유틸 
# ---------------------------
def _apply_photometric_norm(face_bgr):
    # 1) CLAHE in LAB
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L = clahe.apply(L)
    lab = cv2.merge([L, A, B])
    face_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2) Gray-world white balance
    eps = 1e-6
    avg_b, avg_g, avg_r = face_bgr[:,:,0].mean()+eps, face_bgr[:,:,1].mean()+eps, face_bgr[:,:,2].mean()+eps
    gray_mean = (avg_b + avg_g + avg_r) / 3.0
    scale_b, scale_g, scale_r = gray_mean/avg_b, gray_mean/avg_g, gray_mean/avg_r
    face_bgr = face_bgr.astype(np.float32)
    face_bgr[:,:,0] *= scale_b
    face_bgr[:,:,1] *= scale_g
    face_bgr[:,:,2] *= scale_r
    face_bgr = np.clip(face_bgr, 0, 255).astype(np.uint8)

    # 3) Mild gamma correction
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    mid = max(np.median(gray)/255.0, 1e-3)
    gamma = float(np.clip(np.log(0.5)/np.log(mid), 0.67, 1.5))
    inv = 1.0/gamma
    table = (np.linspace(0,1,256) ** inv * 255).astype(np.uint8)
    face_bgr = cv2.LUT(face_bgr, table)
    return face_bgr

def preprocess_face_tensor(face_bgr, use_illum=False):
    if use_illum:
        face_bgr = _apply_photometric_norm(face_bgr)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    transform = xception_default_data_transforms['test']
    return transform(pil_image.fromarray(face_rgb)).unsqueeze(0)

def predict_base(face_bgr, use_illum=False):
    with torch.no_grad():
        input_tensor = preprocess_face_tensor(face_bgr, use_illum=use_illum)
        output = model(input_tensor)
        probs = softmax(output)[0]
        confidence_fake = float(probs[1].item())
        pred = int(torch.argmax(probs).item())  # 0: REAL, 1: FAKE
    return pred, confidence_fake

def predict_tta(face_bgr, use_illum=False):
    augments = [
        lambda x: x,
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.convertScaleAbs(x, alpha=1.0, beta=10),   # brighter
        lambda x: cv2.convertScaleAbs(x, alpha=0.9, beta=-10),  # darker
    ]
    acc = 0.0
    with torch.no_grad():
        for aug in augments:
            img = aug(face_bgr.copy())
            input_tensor = preprocess_face_tensor(img, use_illum=use_illum)
            output = model(input_tensor)
            acc += float(softmax(output)[0][1].item())
    conf = acc / len(augments)
    pred = 1 if conf >= 0.5 else 0
    return pred, conf

def detect_face_bbox(frame_bgr, detector='auto', dnn_conf=0.6):
    """dlib 우선, 실패 시 DNN(있으면) 백업"""
    h, w = frame_bgr.shape[:2]
    # dlib
    if detector in ('auto','dlib'):
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)
            if faces:
                f = faces[0]
                x1,y1,x2,y2 = max(f.left(),0), max(f.top(),0), min(f.right(),w), min(f.bottom(),h)
                if (x2-x1)>0 and (y2-y1)>0:
                    return x1,y1,x2,y2
        except Exception as e:
            print(f"[WARN] dlib detect error: {e}")
    # DNN
    if detector in ('auto','dnn') and face_net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300,300)), 1.0, (300,300), (104,177,123))
        face_net.setInput(blob)
        detections = face_net.forward()
        best, best_conf = None, 0.0
        for i in range(detections.shape[2]):
            conf = float(detections[0,0,i,2])
            if conf > dnn_conf:
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                x1,y1,x2,y2 = box.astype(int)
                x1,y1 = max(x1,0), max(y1,0)
                x2,y2 = min(x2,w), min(y2,h)
                if conf > best_conf and (x2-x1)>0 and (y2-y1)>0:
                    best, best_conf = (x1,y1,x2,y2), conf
        return best
    return None

def encode_jpg_base64(image_bgr):
    ok, buf = cv2.imencode('.jpg', image_bgr)
    if not ok:
        return None
    return base64.b64encode(buf).decode('utf-8')


@app.route('/predict', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    video_file = request.files['file']
    task_id = request.form.get('taskId')
    if not task_id:
        task_id = str(uuid.uuid4())
        print(f"[INFO] taskId 없음 → 새로 생성됨: {task_id}")

    mode = request.form.get('mode', 'default')  # "default" | "precision"
    use_tta = request.form.get('use_tta')
    use_illum = request.form.get('use_illum')
    detector = request.form.get('detector', 'auto')  # auto|dlib|dnn
    smooth_window = int(request.form.get('smooth_window', 0) or 0)
    min_face = int(request.form.get('min_face', 64) or 64)
    sample_count = int(request.form.get('sample_count', 10) or 10)

    # 프리셋
    if mode == 'precision':
        use_tta = True if use_tta is None else (use_tta.lower() == 'true')
        use_illum = True if use_illum is None else (use_illum.lower() == 'true')
        smooth_window = smooth_window or 5
        sample_count = sample_count or 15
    else:
        use_tta = False if use_tta is None else (use_tta.lower() == 'true')
        use_illum = False if use_illum is None else (use_illum.lower() == 'true')
        smooth_window = smooth_window or 0
        sample_count = sample_count or 10


    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_path = tmp.name
        video_file.save(video_path)


    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    
    if num_frames == 0:
        cap.release()
        os.remove(video_path)
        return jsonify({'error': 'Invalid video or zero frames'}), 400


    # 균등 샘플링 (sample_count개)
    step = max(1, num_frames // max(1, sample_count))
    target_indices = set([min(i*step, num_frames-1) for i in range(max(1, sample_count))])

    results = []
    max_confidence = -1.0
    max_conf_frame = None
    processed_frames = 0
    expected = len(target_indices)
    q = deque(maxlen=smooth_window) if smooth_window > 0 else None

    frame_idx = 0

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
                    bbox = detect_face_bbox(frame, detector=detector)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size > 0 and min(face_img.shape[:2]) >= min_face:
                            mean_luma = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY).mean()
                            auto_illum = use_illum or (mean_luma < 25 or mean_luma > 230)

                            if use_tta:
                                pred, confidence = predict_tta(face_img, use_illum=auto_illum)
                            else:
                                pred, confidence = predict_base(face_img, use_illum=auto_illum)

                            if q is not None:
                                q.append(confidence)
                                conf_s = float(sum(q)/len(q))
                                pred_s = 1 if conf_s >= 0.5 else 0
                                results.append({'pred': pred_s, 'confidence': conf_s})
                            else:
                                results.append({'pred': pred, 'confidence': confidence})

                            if confidence > max_confidence:
                                max_confidence = confidence
                                max_conf_frame = face_img.copy()

                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if gray.dtype != 'uint8':
                        gray = gray.astype('uint8')
                    faces = face_detector(gray)
                    if faces:
                        x1 = max(faces[0].left(), 0)
                        y1 = max(faces[0].top(), 0)
                        x2 = min(faces[0].right(), frame.shape[1])
                        y2 = min(faces[0].bottom(), frame.shape[0])
                        face_img = frame[y1:y2, x1:x2]
                        pred, confidence = predict(face_img)
                        results.append({'pred': pred, 'confidence': confidence})

                        if confidence > max_confidence:
                            max_confidence = confidence
                            max_conf_frame = face_img.copy()
            except Exception as e:
                print(f"Error in face detection or prediction: {e}")
        
            # 진행률 계산 및 전송
            processed_frames += 1
            progress_percent = int(100 * processed_frames / max(1, expected))
            send_progress_to_spring(task_id, progress_percent)

        frame_idx+= 1

    cap.release()
    os.remove(video_path)

    if not results:
        return jsonify({
            'result': 'no face detected',
            'options_used': {
                'mode': mode, 'use_tta': use_tta, 'use_illum': use_illum,
                'detector': detector, 'smooth_window': smooth_window,
                'min_face': min_face, 'sample_count': sample_count
            },
            'taskId': task_id
        }), 200

    final_label = 'FAKE' if sum(r['pred'] for r in results) > len(results) // 2 else 'REAL'

    # 가장 높은 confidence 프레임을 이미지로 저장하고 base64로 인코딩
    if max_conf_frame is not None:
        _, buffer = cv2.imencode('.jpg', max_conf_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
    else:
        img_base64 = None

    avg_confidence = sum(r['confidence'] for r in results) / len(results)

    #테스트
    print(f"결과: {final_label}")
    print(f"평균 fake confidence: {avg_confidence:.4f}")
    print(f"최고 confidence: {max_confidence:.4f}")
    #print(f"이미지 image: {img_base64}")

    return jsonify({
        'result': final_label,
        'average_fake_confidence': round(avg_confidence, 4),
        'max_confidence': round(max_confidence, 4) if max_confidence >= 0 else None,
        'most_suspect_image': img_base64,  # base64 encoded image
        'options_used': {
            'mode': mode, 'use_tta': use_tta, 'use_illum': use_illum,
            'detector': detector, 'smooth_window': smooth_window,
            'min_face': min_face, 'sample_count': sample_count
        },
        'taskId': task_id
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
