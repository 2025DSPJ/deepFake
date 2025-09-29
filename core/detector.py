import cv2, numpy as np
import os

# dlib optional
try:
    import dlib
    face_detector = dlib.get_frontal_face_detector()
    dlib_available = True
except Exception:
    face_detector = None
    dlib_available = False

from .config import DNN_PROTO, DNN_MODEL

# OpenCV DNN (있을 때만)
try:
    face_net = (cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
                if (os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL)) else None)
except Exception:
    face_net = None

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
        except Exception:
            pass

    if detector in ('auto','dnn') and face_net is not None:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300,300)), 1.0, (300,300), (104,177,123))
        face_net.setInput(blob)
        dets = face_net.forward()
        for i in range(dets.shape[2]):
            conf = float(dets[0,0,i,2])
            if conf > dnn_conf:
                box = dets[0,0,i,3:7]*np.array([w,h,w,h])
                x1,y1,x2,y2 = box.astype(int)
                x1,y1 = max(x1,0), max(y1,0)
                x2,y2 = min(x2,w), min(y2,h)
                if (x2-x1)>0 and (y2-y1)>0:
                    boxes.append((x1,y1,x2,y2, conf))

    # dedup + area sort
    boxes = list({(x1,y1,x2,y2):conf for x1,y1,x2,y2,conf in boxes}.items())
    boxes = [(b[0][0],b[0][1],b[0][2],b[0][3],b[1]) for b in boxes]
    boxes.sort(key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes[:max_boxes]

def robust_detect(frame, detector='dnn', dnn_conf=0.30, resize_long=720, max_boxes=5):
    H, W = frame.shape[:2]
    # 1차 DNN
    try:
        bboxes = detect_face_bboxes(frame, detector='dnn', dnn_conf=dnn_conf, max_boxes=max_boxes)
    except Exception:
        bboxes = []

    # 리사이즈 후 재시도
    if not bboxes:
        long_side = resize_long or 720
        scale = float(long_side) / float(max(H, W))
        fr = cv2.resize(frame, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        try:
            b2 = detect_face_bboxes(fr, detector='dnn', dnn_conf=0.25, max_boxes=max_boxes)
        except Exception:
            b2 = []
        if b2:
            inv = (1.0/scale) if scale>0 else 1.0
            bboxes = [(int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv), conf) for (x1,y1,x2,y2,conf) in b2]

    # 폴백 dlib
    if not bboxes and detector != 'dlib' and dlib_available and face_detector is not None:
        try:
            fb = detect_face_bboxes(frame, detector='dlib', max_boxes=max_boxes) or []
            bboxes = fb
        except Exception:
            pass
    return bboxes
