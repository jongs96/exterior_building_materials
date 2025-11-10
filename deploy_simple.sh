#!/bin/bash
# AWS EC2 배포 스크립트 (Python 3.10 사용)
# Ubuntu 22.04 LTS 기준

set -e  # 에러 발생 시 스크립트 중단

echo "🚀 Building Material Classifier 배포 시작"
echo "=========================================="

# 1. 시스템 업데이트
echo "📦 시스템 패키지 업데이트..."
sudo apt-get update
sudo apt-get upgrade -y

# 2. 필수 패키지 설치 (Python 3.10은 기본 포함)
echo "📦 필수 패키지 설치..."
sudo apt-get install -y python3 python3-venv python3-pip nginx git libmagic1

# 3. 현재 디렉토리 확인 (스크립트가 있는 위치)
PROJECT_DIR=$(pwd)
echo "📁 프로젝트 디렉토리: $PROJECT_DIR"

# 4. Python 가상환경 생성
echo "🐍 Python 가상환경 생성..."
python3 -m venv venv
source venv/bin/activate

# 5. Python 패키지 설치
echo "📦 Python 패키지 설치..."
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn

# 6. 필요한 디렉토리 생성
echo "📁 디렉토리 생성..."
mkdir -p logs
mkdir -p uploads
mkdir -p static
mkdir -p templates

# 7. 로그 디렉토리 권한 설정
sudo chown -R ubuntu:www-data logs
sudo chown -R ubuntu:www-data uploads
sudo chmod -R 775 logs uploads

# 8. Nginx 설정 (프로젝트 경로 자동 설정)
echo "🌐 Nginx 설정..."
# nginx.conf에서 경로를 현재 프로젝트 경로로 변경
sed "s|/home/ubuntu/building-material-classifier|$PROJECT_DIR|g" nginx.conf | sudo tee /etc/nginx/sites-available/building-material-classifier > /dev/null
sudo ln -sf /etc/nginx/sites-available/building-material-classifier /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

# 9. systemd 서비스 설정 (프로젝트 경로 자동 설정)
echo "⚙️ systemd 서비스 설정..."
# service 파일에서 경로를 현재 프로젝트 경로로 변경
sed "s|/home/ubuntu/building-material-classifier|$PROJECT_DIR|g" building-material-classifier.service | sudo tee /etc/systemd/system/building-material-classifier.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable building-material-classifier
sudo systemctl start building-material-classifier

# 10. 방화벽 설정 (UFW)
echo "🔥 방화벽 설정..."
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# 11. 서비스 상태 확인
echo ""
echo "✅ 배포 완료!"
echo "=========================================="
echo "서비스 상태:"
sudo systemctl status building-material-classifier --no-pager
echo ""
echo "Nginx 상태:"
sudo systemctl status nginx --no-pager
echo ""
echo "🌐 웹사이트 접속: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo ""
echo "📝 유용한 명령어:"
echo "  - 서비스 재시작: sudo systemctl restart building-material-classifier"
echo "  - 로그 확인: sudo journalctl -u building-material-classifier -f"
echo "  - Nginx 로그: sudo tail -f /var/log/nginx/building-material-classifier-error.log"
