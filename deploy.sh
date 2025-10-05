#!/bin/bash
# deploy.sh

docker pull jihye0623/deeptruth-deepfake:latest

# 기존에 실행 중이던 컨테이너가 있다면 중지하고 삭제
if [ "$(docker ps -q -f name=deepfake_container)" ]; then
    docker stop deepfake_container
    docker rm deepfake_container
fi

# 새로운 이미지로 컨테이너를 실행
docker run -d --gpus all \
  -p 8000:5001 \
  --name deepfake_container \
  --restart unless-stopped \
  --env-file ./deepFake/.env \
  jihye0623/deeptruth-deepfake:latest