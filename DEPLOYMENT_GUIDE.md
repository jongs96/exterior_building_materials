# 🚀 AWS EC2 배포 가이드

건축 외장재 분류기를 AWS EC2에 배포하는 전체 가이드입니다.

## 📋 사전 준비

### 필요한 것
- AWS 계정
- SSH 클라이언트 (Windows: PuTTY, Mac/Linux: 터미널)
- 프로젝트 파일
- 모델 파일 (`building_material_classifier_pytorch.pth`)

---

## 1️⃣ AWS EC2 인스턴스 생성

### 1.1 EC2 인스턴스 설정
1. AWS 콘솔 → EC2 → "인스턴스 시작" 클릭
2. **이름**: `building-material-classifier`
3. **AMI**: Ubuntu Server 22.04 LTS
4. **인스턴스 유형**: `t3.medium` (권장) 또는 `t3.small` (최소)
   - vCPU: 2개
   - 메모리: 4GB (PyTorch 모델 로딩에 필요)
5. **키 페어**: 새로 생성하거나 기존 키 선택 (`.pem` 파일 다운로드)
6. **스토리지**: 20GB (gp3)

### 1.2 보안 그룹 설정
다음 포트를 오픈:
- **SSH**: 포트 22 (내 IP만 허용 권장)
- **HTTP**: 포트 80 (0.0.0.0/0)
- **HTTPS**: 포트 443 (0.0.0.0/0) - 선택사항

### 1.3 Elastic IP 할당 (선택사항)
- 고정 IP가 필요한 경우 Elastic IP 할당

---

## 2️⃣ 서버 접속

### Windows (PuTTY)
```bash
# PuTTYgen으로 .pem을 .ppk로 변환
# PuTTY에서 접속: ubuntu@<EC2-PUBLIC-IP>
```

### Mac/Linux
```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

---

## 3️⃣ 프로젝트 파일 업로드

### 방법 1: SCP 사용 (권장)
```bash
# 로컬 컴퓨터에서 실행
scp -i your-key.pem -r C:\github\picture_machine\dataset ubuntu@<EC2-PUBLIC-IP>:/home/ubuntu/building-material-classifier
```

### 방법 2: Git 사용
```bash
# EC2 서버에서 실행
cd /home/ubuntu
git clone <your-repo-url> building-material-classifier
cd building-material-classifier
```

### 방법 3: FileZilla (GUI)
1. FileZilla 설치
2. SFTP 연결 설정
3. 파일 드래그 앤 드롭

---

## 4️⃣ 배포 스크립트 실행

### 4.1 배포 스크립트 실행 권한 부여
```bash
cd /home/ubuntu/building-material-classifier
chmod +x deploy.sh
```

### 4.2 배포 실행
```bash
./deploy.sh
```

이 스크립트는 자동으로:
- 시스템 패키지 업데이트
- Python 3.11 및 필수 패키지 설치
- 가상환경 생성 및 Python 패키지 설치
- Nginx 설정
- systemd 서비스 설정
- 방화벽 설정

**예상 소요 시간**: 5-10분

---

## 5️⃣ 서비스 확인

### 5.1 서비스 상태 확인
```bash
# Flask 앱 상태
sudo systemctl status building-material-classifier

# Nginx 상태
sudo systemctl status nginx
```

### 5.2 로그 확인
```bash
# Flask 앱 로그 (실시간)
sudo journalctl -u building-material-classifier -f

# Nginx 에러 로그
sudo tail -f /var/log/nginx/building-material-classifier-error.log

# 앱 로그
tail -f /home/ubuntu/building-material-classifier/logs/error.log
```

### 5.3 웹사이트 접속
```
http://<EC2-PUBLIC-IP>
```

---

## 6️⃣ 유용한 명령어

### 서비스 관리
```bash
# 서비스 시작
sudo systemctl start building-material-classifier

# 서비스 중지
sudo systemctl stop building-material-classifier

# 서비스 재시작
sudo systemctl restart building-material-classifier

# 서비스 자동 시작 활성화
sudo systemctl enable building-material-classifier
```

### 코드 업데이트
```bash
cd /home/ubuntu/building-material-classifier

# Git으로 최신 코드 가져오기
git pull

# 또는 파일 직접 업로드 후

# 서비스 재시작
sudo systemctl restart building-material-classifier
```

### Nginx 관리
```bash
# Nginx 설정 테스트
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx

# Nginx 로그 확인
sudo tail -f /var/log/nginx/building-material-classifier-access.log
```

---

## 7️⃣ 문제 해결

### 서비스가 시작되지 않는 경우
```bash
# 로그 확인
sudo journalctl -u building-material-classifier -n 50

# 수동으로 실행해보기
cd /home/ubuntu/building-material-classifier
source venv/bin/activate
gunicorn -c gunicorn_config.py "app:create_app()"
```

### 502 Bad Gateway 에러
```bash
# Flask 앱이 실행 중인지 확인
sudo systemctl status building-material-classifier

# 포트 8000이 열려있는지 확인
sudo netstat -tlnp | grep 8000
```

### 메모리 부족
```bash
# 메모리 사용량 확인
free -h

# 스왑 메모리 추가 (필요시)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 8️⃣ HTTPS 설정 (선택사항)

### Let's Encrypt SSL 인증서 설치
```bash
# Certbot 설치
sudo apt-get install certbot python3-certbot-nginx

# SSL 인증서 발급 (도메인 필요)
sudo certbot --nginx -d your-domain.com

# 자동 갱신 테스트
sudo certbot renew --dry-run
```

---

## 9️⃣ 모니터링 (선택사항)

### CloudWatch 설정
1. EC2 인스턴스에 CloudWatch 에이전트 설치
2. 메트릭 수집 설정
3. 알람 설정 (CPU, 메모리, 디스크)

### 간단한 모니터링
```bash
# CPU 및 메모리 사용량
htop

# 디스크 사용량
df -h

# 네트워크 연결
sudo netstat -tulpn
```

---

## 🎉 배포 완료!

웹사이트가 정상적으로 작동하면 배포 완료입니다!

**접속 URL**: `http://<EC2-PUBLIC-IP>`

---

## 📞 지원

문제가 발생하면:
1. 로그 확인
2. 서비스 상태 확인
3. 방화벽 및 보안 그룹 확인
4. 메모리 및 디스크 공간 확인
