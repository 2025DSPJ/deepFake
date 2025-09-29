import cv2, numpy as np
from PIL import Image as pil_image
from dataset.transform import xception_default_data_transforms

def to_tensor_bgr(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    transform = xception_default_data_transforms['test']
    return transform(pil_image.fromarray(face_rgb)).unsqueeze(0)

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
