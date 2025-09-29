import os, cv2, base64, requests, time
import numpy as np
from .config import SPRING_SERVER_URL

def send_progress_to_spring(task_id, percent, login_id):
    try:
        payload = {'taskId': task_id, 'progress': percent, 'loginId': login_id}
        requests.post(SPRING_SERVER_URL, json=payload, headers={'Content-Type':'application/json'}, timeout=1)
    except Exception:
        pass

def encode_jpg_base64(image_bgr):
    ok, buf = cv2.imencode('.jpg', image_bgr)
    if not ok: return None
    return base64.b64encode(buf).decode('utf-8')

def uniform_indices(num_frames, sample_count):
    step = max(1, num_frames // max(1, sample_count))
    return set([min(i*step, num_frames-1) for i in range(max(1, sample_count))])

def shrink_value_scalar(x, alpha=1.2):
    x = float(np.clip(x, 0.0, 1.0)); return x ** alpha

def shrink_value_vec(arr, alpha=1.2):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return np.power(arr, alpha, dtype=np.float32)
