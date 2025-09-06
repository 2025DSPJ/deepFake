# 1. 베이스 이미지: PyTorch와 CUDA가 포함된 공식 이미지 사용
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# 2. 시스템 의존성 설치 (dlib, OpenCV 등)
# noninteractive 설정을 통해 설치 중 묻는 창이 뜨지 않도록 함
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 파이썬 의존성 설치 (requirements.txt 먼저 복사하여 캐시 활용)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트 파일 전체 복사
COPY . .

# 6. 포트 노출
# app.py 마지막 줄의 port=5001과 일치
EXPOSE 5001

# 7. 서버 실행
# gunicorn을 사용하여 안정적으로 앱 실행. 0.0.0.0으로 바인딩해야 외부 접속 가능
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "300", "app:app"]
