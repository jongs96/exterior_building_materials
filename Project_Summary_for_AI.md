# 딥러닝 기반 건축 자재 분류 서비스: AI 요약 보고서

## 1. 프로젝트 개요

*   **프로젝트명:** 딥러닝 기반 건축 자재 이미지 분류 웹 서비스
*   **핵심 목표:** 사용자가 업로드한 이미지를 분석하여 건축 자재의 종류를 자동으로 식별하고 알려주는 AI 솔루션 구축
*   **주요 기능:**
    *   웹 인터페이스를 통한 간편한 이미지 업로드
    *   PyTorch 딥러닝 모델(ResNet18)을 이용한 실시간 이미지 분류
    *   분류 결과(자재명, 신뢰도)를 사용자에게 즉시 제공
*   **기대 효과:**
    *   건설 현장 및 관련 산업에서의 업무 효율성 증대
    *   자재 오분류로 인한 인적 오류 및 비용 감소
    *   자재 재활용 분류 자동화에 기여

## 2. 기술 스택 (Technology Stack)

*   **백엔드 (Backend):**
    *   언어: Python 3
    *   프레임워크: Flask
*   **AI / 머신러닝 (AI / Machine Learning):**
    *   메인 라이브러리: PyTorch
    *   보조 라이브러리: Torchvision, Scikit-learn
*   **프론트엔드 (Frontend):**
    *   HTML5, CSS3, JavaScript
*   **서버 및 배포 (Server & Deployment):**
    *   웹 서버: Nginx
    *   WAS(Web Server Gateway Interface): Gunicorn
*   **개발 도구:**
    *   Git, Visual Studio Code, venv

## 3. 시스템 아키텍처 (System Architecture)

### 3.1. 애플리케이션 아키텍처 (Application Architecture)

*   **기본 구조:** Flask 애플리케이션 팩토리 패턴
    *   **설명:** 애플리케이션의 생성 및 설정을 중앙에서 관리하여 유연성과 확장성을 확보하는 디자인 패턴입니다.
*   **모듈 분리: 블루프린트 (Blueprints)**
    *   `main` 블루프린트: 사용자가 보는 웹 페이지(HTML)를 렌더링합니다.
    *   `api` 블루프린트: 모델 추론을 위한 RESTful API 엔드포인트(`/api/predict`)를 제공합니다.
*   **서비스 계층 (Service Layer)**
    *   `ImageService`: 이미지 리사이징, 정규화 등 모델 입력에 필요한 전처리 로직을 담당합니다.
    *   `ModelService`: 사전 훈련된 PyTorch 모델(`.pth`)을 로드하고, 입력된 이미지 텐서에 대한 추론(prediction)을 수행합니다.

### 3.2. 배포 아키텍처 (Deployment Architecture)

`Client (Web Browser) <-> Nginx (Web Server) <-> Gunicorn (WSGI) <-> Flask Application`

*   **흐름 설명:**
    1.  Nginx가 외부로부터의 HTTP 요청을 가장 먼저 받습니다. (정적 파일 처리 및 로드 밸런싱)
    2.  Nginx는 동적 요청을 Gunicorn에 전달합니다.
    3.  Gunicorn은 여러 개의 Flask 애플리케이션 프로세스를 관리하며 요청을 분산 처리합니다.
    4.  Flask 애플리케이션이 최종적으로 요청을 처리하고 응답을 생성합니다.

## 4. 워크플로우 (Workflow)

### 4.1. 모델 훈련 워크플로우

1.  **데이터 수집:** 웹 크롤러(`google_image_crawler.py`)를 이용해 클래스별(벽돌, 콘크리트 등) 건축 자재 이미지 수집.
2.  **데이터 전처리:** 모든 이미지의 크기를 224x224 픽셀로 통일하고 정규화(Normalization) 수행.
3.  **데이터 증강 (Augmentation):** 훈련 데이터에 한해 무작위 회전, 밝기 조절 등을 적용하여 모델의 강건성(robustness) 확보.
4.  **모델 선정:** 사전 훈련된 ResNet18 모델을 '전이 학습(Transfer Learning)' 방식으로 활용.
5.  **모델 커스터마이징:** 마지막 분류기(classifier) 레이어를 프로젝트의 클래스 수에 맞게 교체.
6.  **모델 훈련:** 교체된 레이어를 중심으로 모델을 재훈련(`fine-tuning`)하고, 검증 데이터셋으로 성능 평가.
7.  **모델 저장:** 가장 성능이 좋은 모델의 가중치를 `.pth` 파일로 저장.

### 4.2. 서비스 추론 플로우차트 (Inference Flowchart)

1.  **[사용자]** 웹 브라우저에서 건축 자재 이미지를 선택하고 '분석' 버튼 클릭.
2.  **[프론트엔드]** JavaScript가 이미지를 포함하여 서버의 `/api/predict` 엔드포인트로 `POST` 요청 전송.
3.  **[백엔드 - Flask]** `api` 블루프린트의 라우트 함수가 요청을 수신.
4.  **[백엔드 - ImageService]** 수신된 이미지를 가져와 PyTorch 텐서로 변환 (리사이징, 정규화 등 전처리 수행).
5.  **[백엔드 - ModelService]** 전처리된 텐서를 입력으로 받아 메모리에 로드된 ResNet18 모델을 통해 추론 실행.
6.  **[백엔드 - ModelService]** 모델의 출력(각 클래스에 대한 확률 값)을 분석하여 가장 높은 확률의 클래스 이름과 신뢰도를 결정.
7.  **[백엔드 - Flask]** 추론 결과를 `{ "class_name": "벽돌", "confidence": 0.98 }`과 같은 JSON 형식으로 변환하여 응답.
8.  **[프론트엔드]** JavaScript가 서버로부터 받은 JSON 응답을 파싱하여 웹 페이지의 특정 영역에 결과를 동적으로 표시.
9.  **[사용자]** 자신의 이미지에 대한 분석 결과를 웹 페이지에서 확인.
