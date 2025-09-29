import cv2, numpy as np, torch
from .model import get_model_and_softmax
from .preprocess import to_tensor_bgr, need_illum, illum_hsv

# ---- aug set ----
def aug_gamma(img, gamma=1.05):
    inv = 1.0/max(gamma,1e-6)
    lut = (np.linspace(0,1,256)**inv*255).astype(np.uint8)
    return cv2.LUT(img, lut)

def aug_hsv_jitter(img, dh=3, s_scale=1.03, v_scale=1.03):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    h = ((h.astype(np.int16)+dh)%180).astype(np.uint8)
    s = np.clip(s.astype(np.float32)*s_scale,0,255).astype(np.uint8)
    v = np.clip(v.astype(np.float32)*v_scale,0,255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)

def aug_gaussian_blur(img, sigma=0.6):
    k = max(3, int(2*round(3*sigma)+1))
    return cv2.GaussianBlur(img, (k,k), sigmaX=sigma)

def aug_unsharp(img, amount=1.2, sigma=1.0):
    blur = cv2.GaussianBlur(img, (0,0), sigma)
    return cv2.addWeighted(img, amount, blur, -(amount-1.0), 0)

# ---- basic predict ----
def predict_single_tensor(img_tensor):
    model, softmax = get_model_and_softmax()
    with torch.no_grad():
        output = model(img_tensor)
        probs = softmax(output)[0]
        conf_fake = float(probs[1].item())
        pred = int(torch.argmax(probs).item())  # 0: REAL, 1: FAKE
    return pred, conf_fake

def predict(face_bgr):
    return predict_single_tensor(to_tensor_bgr(face_bgr))

# ---- TTA ----
def predict_tta_weighted(face_bgr, use_cond_illum=False):
    proc = face_bgr.copy()
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    _, conf_raw = predict_single_tensor(to_tensor_bgr(proc))
    if use_cond_illum and need_illum(gray):
        proc = illum_hsv(proc)

    # gating
    if not (0.4 <= conf_raw <= 0.9 or abs(conf_raw-0.6) <= 0.03):
        return (1 if conf_raw >= 0.5 else 0), float(conf_raw)

    augs = [
        ("orig",  lambda x:x,                           0.50),
        ("hsv",   lambda x:aug_hsv_jitter(x,-3,0.97,0.97), 0.10),
        ("gblur", lambda x:aug_gaussian_blur(x,0.5),    0.13),
        ("usm",   lambda x:aug_unsharp(x,1.2,1.0),      0.08),
        ("gamma", lambda x:aug_gamma(x,1.05),           0.20),
    ]
    w_sum = sum(w for _,_,w in augs)
    augs = [(n,f,w/w_sum) for (n,f,w) in augs]

    vals, ws = [], []
    with torch.no_grad():
        for _, aug, w in augs:
            img = aug(proc.copy())
            _, conf_i = predict_single_tensor(to_tensor_bgr(img))
            vals.append(float(conf_i)); ws.append(float(w))
    confs = np.array(vals, dtype=np.float32)
    base_ws = np.array(ws, dtype=np.float32)

    delta = confs - float(conf_raw)
    k_push = 6.0
    w_adj = base_ws * np.exp(-k_push * np.maximum(0.0, delta))
    w_final = w_adj / max(1e-12, w_adj.sum())

    conf_tta = float((w_final * confs).sum())
    std_conf = float(np.std(confs))
    if std_conf < 0.01: conf_tta = conf_raw
    if std_conf > 0.05:
        shrink = min(0.25, 5.0*(std_conf-0.05))
        conf_tta = 0.5 + (conf_tta-0.5)*(1.0-shrink)

    max_drift = 0.02 + 0.30*std_conf
    drift = conf_tta - conf_raw
    if abs(drift) > max_drift:
        conf_tta = conf_raw + np.sign(drift)*max_drift

    return (1 if conf_tta >= 0.5 else 0), float(conf_tta)

def infer_prob(face_bgr, mode, use_tta, use_illum):
    if (mode == 'precision') and use_tta:
        _, p = predict_tta_weighted(face_bgr, use_cond_illum=bool(use_illum))
    else:
        _, p = predict(face_bgr)
    return float(p)
