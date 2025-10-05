// Jenkinsfile
pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = 'dockerhub-credentials'
        DOCKERHUB_USERNAME = 'jihye0623' // Docker Hub ID
        IMAGE_NAME = "${DOCKERHUB_USERNAME}/deeptruth-deepfake"
        DEPLOY_SERVER_IP = credentials('deploy-server-ip')
        DEPLOY_SERVER_CREDENTIALS = 'deeptruth-server-key'
    }

    stages {
        stage('Checkout') {
            steps {
                // 1. GitHub에서 최신 코드를 가져옴
                git branch: 'main', url: 'https://github.com/25-HF003/deepFake.git'
            }
        }

        stage('Build Image') {
            steps {
                // 2. Dockerfile을 이용해 이미지를 빌드
                script {
                    docker.withRegistry('https://registry.hub.docker.com', DOCKERHUB_CREDENTIALS) {
                        def customImage = docker.build(IMAGE_NAME, "--build-arg BUILDKIT_INLINE_CACHE=1 .")
                        
                        // 3. 빌드된 이미지를 Docker Hub에 푸시
                        customImage.push("${env.BUILD_NUMBER}")
                        customImage.push("latest")
                    }
                }
            }
        }

        stage('Deploy to Server') {
            steps {
                // 4. SSH로 딥페이크 서버에 접속하여 배포 스크립트 실행
                sshagent(credentials: [DEPLOY_SERVER_CREDENTIALS]) {
                    sh "ssh -o StrictHostKeyChecking=no ubuntu@${DEPLOY_SERVER_IP} 'bash -s' < ./deploy.sh"
                }
            }
        }
    }
}