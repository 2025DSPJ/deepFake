import os, cv2, uuid, time, tempfile
import numpy as np

from .config import OUTPUT_ROOT, TARGET_FPS_DEFAULT, MAX_LAT_MS_DEFAULT
from .detector import robust_detect, dlib_available, face_detector, face_net
from .preprocess import need_illum
from .infer import predict, predict_tta_weighted
from .metrics import var_laplace, luma_mean, tta_consistency_std, temporal_delta_stats
from .aggregate import aggregate_and_decide
from .vis import save_heatmap
from .utils import send_progress_to_spring, encode_jpg_base64, uniform_indices, shrink_value_scalar, shrink_value_vec

def run_pipeline(*, files, form):
    if 'file' not in files:
        return {'error':'No file uploaded'}, 400

    start_ts = time.time()
    video_file = files['file']
    task_id = form.get('taskId') or str(uuid.uuid4())
    login_id = form.get('loginId')
    mode = form.get('mode', 'default')
    use_tta = form.get('use_tta')
    use_illum = form.get('use_illum')
    detector = form.get('detector') or ('dnn' if mode=='precision' else 'auto')
    smooth_window = int(form.get('smooth_window', 0) or 0)
    min_face = int(form.get('min_face', 96 if mode=='precision' else 64) or (96 if mode=='precision' else 64))
    sample_count = int(form.get('sample_count', 20 if mode=='precision' else 10) or (20 if mode=='precision' else 10))

    # 속도목표
    target_fps = float(form.get('target_fps', TARGET_FPS_DEFAULT))
    max_latency_ms = float(form.get('max_latency_ms', MAX_LAT_MS_DEFAULT))
    target_fps = 0.27     # 기본모드 목표 처리량
    max_latency_ms = 4000.0  # 기본모드 지연 목표

    if mode == 'precision':
        use_tta   = True if use_tta is None else (use_tta.lower() == 'true')
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

    # 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_path = tmp.name
        try: video_file.stream.seek(0)
        except Exception: pass
        video_file.save(tmp)

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames == 0:
        cap.release(); os.remove(video_path)
        return {'error':'Invalid video or zero frames'}, 400

    # 균등 샘플링
    target_indices = uniform_indices(num_frames, sample_count)

    results = []
    per_frame_conf = []
    raw_conf_for_vote = []
    scene_lumas = []
    ema = None

    max_confidence = -1.0
    max_conf_frame = None
    processed_frames = 0
    expected = len(target_indices)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None: break
        if frame_idx in target_indices:
            try:
                if frame.dtype != 'uint8':
                    frame = frame.astype('uint8')

                if mode == 'precision':
                    bboxes = robust_detect(frame, detector=detector)
                    # 중앙 폴백(옵션): BYPASS_DETECT=1
                    if (not bboxes) and os.getenv("BYPASS_DETECT","0") == "1":
                        H,W = frame.shape[:2]; cx,cy = W//2, H//2; s = min(H,W)//3
                        bboxes = [(cx-s,cy-s,cx+s,cy+s, 1.0)]

                    if bboxes:
                        # 상위 2개 후보: 면적 최대 + 신뢰 최대
                        b_area = sorted(bboxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)
                        b_conf = sorted(bboxes, key=lambda b:b[4], reverse=True)
                        chosen = [b_area[0]]
                        if b_conf[0] != b_area[0] and len(b_area)>1: chosen.append(b_conf[0])

                        for (x1,y1,x2,y2,_) in chosen[:2]:
                            face_img = frame[y1:y2, x1:x2]
                            if face_img.size==0 or min(face_img.shape[:2]) < min_face: continue
                            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                            q = var_laplace(gray_face); l = luma_mean(gray_face)

                            pred_i, conf_i = (predict_tta_weighted(face_img, use_cond_illum=use_illum)
                                              if use_tta else predict(face_img))

                            ema = conf_i if ema is None else (0.5*conf_i + 0.5*ema)
                            conf_s = float(ema)

                            raw_conf_for_vote.append((conf_i, q, l))
                            per_frame_conf.append(conf_s)
                            results.append({'pred': 1 if conf_s>=0.5 else 0, 'confidence': conf_s})

                            if conf_i > max_confidence:
                                max_confidence = conf_i; max_conf_frame = face_img.copy()
                            scene_lumas.append(l)
                else:
                    # 기본 모드: dlib 우선
                    if dlib_available and face_detector is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_detector(gray)
                        if faces:
                            f = faces[0]
                            x1,y1,x2,y2 = max(f.left(),0), max(f.top(),0), min(f.right(),frame.shape[1]), min(f.bottom(),frame.shape[0])
                            face_img = frame[y1:y2, x1:x2]
                            if face_img.size>0 and min(face_img.shape[:2])>=64:
                                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                                q = var_laplace(gray_face); l = luma_mean(gray_face)
                                pred_i, conf_i = predict(face_img)
                                ema = conf_i if ema is None else (0.5*conf_i + 0.5*ema)
                                conf_s = float(ema)
                                raw_conf_for_vote.append((conf_i, q, l))
                                per_frame_conf.append(conf_s)
                                results.append({'pred': 1 if conf_s>=0.5 else 0, 'confidence': conf_s})
                                if conf_i > max_confidence:
                                    max_confidence = conf_i; max_conf_frame = face_img.copy()
                                scene_lumas.append(l)
            except Exception as e:
                print(f"[WARN] frame err: {e}")

            processed_frames += 1
            pct = int(100 * processed_frames / max(1, expected))
            send_progress_to_spring(task_id, pct, login_id)
        frame_idx += 1

    cap.release()
    os.remove(video_path)

    if not results:
        return {
            'result': 'no face detected',
            'mode': mode,
            'use_tta': use_tta,
            'use_illum': use_illum,
            'detector': detector,
            'min_face': min_face,
            'sample_count': sample_count,
            'taskId': task_id
        }, 200

    # 집계/판정
    final_label, S, tau_high, frame_vote_ratio = aggregate_and_decide(raw_conf_for_vote, results, scene_lumas, mode)

    processing_time_sec = time.time() - start_ts
    fps_processed = (len(per_frame_conf)/processing_time_sec) if processing_time_sec > 0 else None
    ms_per_sample = ((processing_time_sec / len(per_frame_conf)) * 1000.0) if len(per_frame_conf) > 0 else None
    speed_ok = ((fps_processed or 0.0) >= float(TARGET_FPS_DEFAULT) and (ms_per_sample or 1e12) <= float(MAX_LAT_MS_DEFAULT))

    # 통계(확률 완만 축소)
    conf_arr = np.array(per_frame_conf, dtype=np.float32)
    if conf_arr.size >= 10:
        k_trim = max(1, int(0.10 * conf_arr.size))
        conf_sorted = np.sort(conf_arr)
        conf_trimmed = conf_sorted[:conf_arr.size - k_trim] if (2*k_trim) < conf_sorted.size else conf_sorted
    else:
        conf_trimmed = conf_arr
    conf_trimmed_shr = shrink_value_vec(conf_trimmed, alpha=1.2)
    avg_confidence = float(conf_trimmed_shr.mean()) if conf_trimmed_shr.size else 0.0
    max_conf_s = float(conf_trimmed_shr.max()) if conf_trimmed_shr.size else None
    median_confidence = float(np.median(conf_arr)) if conf_arr.size else None
    variance_confidence = float(conf_arr.var()) if conf_arr.size else None

    # 히트맵
    task_dir = os.path.join(OUTPUT_ROOT, task_id)
    heatmap_path = os.path.join(task_dir, f"heatmap_{task_id}.png")
    per_frame_conf_shr = [float(shrink_value_scalar(x, alpha=1.2)) for x in per_frame_conf]
    try:
        save_heatmap(per_frame_conf_shr, heatmap_path)
    except Exception:
        heatmap_path = None

    # 안정성 지표
    tta_std = tta_mean = temporal_mean = temporal_std = None
    if max_conf_frame is not None:
        try:
            tta_std, tta_mean = tta_consistency_std(max_conf_frame, mode, use_tta, use_illum)
        except Exception: pass
        try:
            temporal_mean, temporal_std = temporal_delta_stats(per_frame_conf)
        except Exception: pass

    img_base64 = encode_jpg_base64(max_conf_frame) if max_conf_frame is not None else None

    return {
        'taskId': task_id,
        'result': final_label,
        'most_suspect_image': img_base64,

        'score_weighted': round(S, 4),
        'threshold_tau': tau_high,
        'frame_vote_ratio': round(frame_vote_ratio, 4),

        'average_fake_confidence': round(avg_confidence, 4),
        'median_confidence': round(median_confidence, 4) if median_confidence is not None else None,
        'variance_confidence': round(variance_confidence, 6) if variance_confidence is not None else None,
        'max_confidence': round(max_conf_s, 4) if (max_conf_s is not None) else None,

        'frames_processed': len(per_frame_conf),
        'processing_time_sec': round(processing_time_sec, 3),

        'mode': mode, 'use_tta': use_tta, 'use_illum': use_illum,
        'detector': detector, 'min_face': min_face, 'sample_count': sample_count,

        'timeseries': {
            'per_frame_conf': [float(x) for x in per_frame_conf],
            'vmin': 0.0, 'vmax': 1.0
        },
        'stability_evidence': {
            'tta_std': round(tta_std, 6) if tta_std is not None else None,
            'tta_mean': round(tta_mean, 6) if tta_mean is not None else None,
            'temporal_delta_mean': round(temporal_mean, 6) if temporal_mean is not None else None,
            'temporal_delta_std': round(temporal_std, 6) if temporal_std is not None else None
        },
        'speed': {
            'fps_processed': round(fps_processed, 3) if fps_processed is not None else None,
            'ms_per_sample': round(ms_per_sample, 1) if ms_per_sample is not None else None,
            'target_fps': float(TARGET_FPS_DEFAULT),
            'max_latency_ms': float(MAX_LAT_MS_DEFAULT),
            'speed_ok': bool(speed_ok),
        }
    }, 200
