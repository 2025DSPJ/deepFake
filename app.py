from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image as pil_image
import cv2
import dlib
import tempfile
import os
from tqdm import tqdm
from network.models import model_selection  # xception 모델 불러오는 함수
from dataset.transform import xception_default_data_transforms
import gdown

# 모델 경로 지정
model_path = './model/xception.pth'

# 모델이 없을 경우, Google Drive에서 다운로드
if not os.path.exists(model_path):
    print("모델이 없어서 Google Drive에서 다운로드")
    os.makedirs('./model', exist_ok=True)
    gdown.download(id='1j8AesqDjbSG0RfqaYaHdfGcVVkpdIPKJ', output=model_path, quiet=False)

app = Flask(__name__)
face_detector = dlib.get_frontal_face_detector()

# 모델 불러오기
model = model_selection(modelname='xception', num_out_classes=2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = xception_default_data_transforms['test']
    img_tensor = transform(pil_image.fromarray(image)).unsqueeze(0)
    return img_tensor

def predict(image):
    with torch.no_grad():
        input_tensor = preprocess_image(image)
        output = model(input_tensor)
        pred = torch.argmax(nn.Softmax(dim=1)(output), dim=1).item()
    return pred

@app.route('/predict', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    video_file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_path = tmp.name
        video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, num_frames // 10)

    predictions = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_num % frame_interval == 0:
            try:
                if frame.dtype != 'uint8':
                    frame = frame.astype('uint8')
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
                    pred = predict(face_img)
                    predictions.append(pred)
            except Exception as e:
                print(f"Error in face detection or prediction: {e}")
        frame_num += 1

    cap.release()
    os.remove(video_path)

    if not predictions:
        return jsonify({'result': 'no face detected'}), 200
    fake_count = predictions.count(1)
    total_count = len(predictions)

    fake_confidence = fake_count / total_count 
    final_result = 'FAKE' if fake_confidence > 0.5 else 'REAL'

    return jsonify({
        'result': final_result,
        'confidence': round(fake_confidence, 2)  * 100
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
