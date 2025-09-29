import cv2, numpy as np
from .infer import infer_prob
from .infer import aug_hsv_jitter, aug_gaussian_blur, aug_unsharp, aug_gamma

def var_laplace(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def luma_mean(gray):
    return float(gray.mean())

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
