# 건축 외장 마감재 분류 웹 어플리케이션 요구사항

## 소개

건축 외장 마감재(벽돌/조적, 스타코, 금속 패널, 석재, 목재 사이딩) 이미지를 자동으로 분류하는 PyTorch 기반 AI 웹 어플리케이션입니다. ResNet50 전이학습 모델을 사용하여 사용자가 업로드한 건축 외장재 이미지를 분석하고, 분류 결과와 함께 해당 외장재에 대한 설명을 제공합니다.

## 용어 정의

- **Building_Material_Classifier**: 건축 외장 마감재 분류 웹 어플리케이션 시스템
- **User**: 건축 외장재 분류 서비스를 이용하는 사용자
- **Material_Image**: 분류 대상이 되는 건축 외장 마감재 이미지
- **Classification_Model**: PyTorch ResNet50 기반 이미지 분류 모델
- **Material_Category**: 5가지 외장재 분류 (brick, metal, stone, stucco, wood)
- **Training_Dataset**: 모델 학습을 위한 라벨링된 이미지 데이터셋 (총 931개)
- **Web_Interface**: Flask 기반 웹 인터페이스

## 요구사항

### 요구사항 1: 데이터 관리

**사용자 스토리:** 개발자로서, 수집된 건축 외장재 이미지 데이터를 효율적으로 관리하고 필요시 재학습할 수 있어야 합니다.

#### 승인 기준

1. THE Building_Material_Classifier SHALL 5가지 외장재 카테고리별로 이미지를 data/raw/ 폴더에 저장한다
2. THE Building_Material_Classifier SHALL 각 카테고리별 이미지 개수를 추적하고 관리한다 (brick: 255, metal: 192, stone: 205, stucco: 133, wood: 146)
3. THE Building_Material_Classifier SHALL 데이터 재분류 스크립트를 통해 잘못 분류된 이미지를 올바른 폴더로 이동할 수 있다
4. THE Building_Material_Classifier SHALL 224x224 픽셀 크기로 이미지를 전처리한다

### 요구사항 2: PyTorch 모델 관리

**사용자 스토리:** 개발자로서, PyTorch 기반 ResNet50 모델을 학습하고 관리할 수 있어야 합니다.

#### 승인 기준

1. THE Building_Material_Classifier SHALL PyTorch ResNet50 전이학습 모델을 사용한다
2. THE Building_Material_Classifier SHALL 훈련/검증 데이터를 80/20 비율로 분할한다
3. THE Building_Material_Classifier SHALL 학습된 모델을 .pth 형식으로 저장한다
4. THE Building_Material_Classifier SHALL 검증 데이터셋에서 최소 65% 이상의 정확도를 달성한다
5. THE Building_Material_Classifier SHALL ImageNet 정규화 파라미터를 사용하여 이미지를 전처리한다

### 요구사항 3: 웹 인터페이스

**사용자 스토리:** 사용자로서, 직관적인 웹 인터페이스를 통해 건축 외장재 이미지를 업로드하고 분류 결과를 확인하고 싶습니다.

#### 승인 기준

1. THE Building_Material_Classifier SHALL 사용자가 이미지 파일을 업로드할 수 있는 인터페이스를 제공한다
2. THE Building_Material_Classifier SHALL 사용자가 카메라를 통해 실시간으로 사진을 촬영할 수 있는 기능을 제공한다
3. WHEN 사용자가 이미지를 업로드하면, THE Building_Material_Classifier SHALL 5초 이내에 분류 결과를 표시한다
4. THE Building_Material_Classifier SHALL 분류 결과와 함께 예측 신뢰도를 백분율로 표시한다
5. THE Building_Material_Classifier SHALL 각 외장재 종류에 대한 설명과 특징을 함께 제공한다

### 요구사항 4: 이미지 처리 및 분류

**사용자 스토리:** 사용자로서, 업로드한 이미지가 정확하게 분석되어 올바른 외장재 종류로 분류되기를 원합니다.

#### 승인 기준

1. WHEN 사용자가 이미지를 업로드하면, THE Building_Material_Classifier SHALL 이미지를 224x224 크기로 전처리한다
2. THE Building_Material_Classifier SHALL PyTorch 모델에 이미지를 전달하여 분류를 수행한다
3. THE Building_Material_Classifier SHALL 5가지 Material_Category 중 가장 높은 확률을 가진 결과를 반환한다
4. THE Building_Material_Classifier SHALL 모든 클래스별 확률 정보를 함께 제공한다
5. THE Building_Material_Classifier SHALL JPG, PNG, WEBP 형식의 이미지 파일을 지원한다

### 요구사항 5: 시스템 성능 및 안정성

**사용자 스토리:** 사용자로서, 안정적이고 빠른 서비스를 이용하고 싶습니다.

#### 승인 기준

1. THE Building_Material_Classifier SHALL 업로드 가능한 이미지 크기를 10MB로 제한한다
2. WHEN 시스템 오류가 발생하면, THE Building_Material_Classifier SHALL 사용자에게 명확한 오류 메시지를 표시한다
3. THE Building_Material_Classifier SHALL 업로드된 이미지를 서버에 영구 저장하지 않고 처리 후 삭제한다
4. THE Building_Material_Classifier SHALL 모바일 기기에서도 정상적으로 작동한다
5. THE Building_Material_Classifier SHALL CPU 환경에서도 정상적으로 모델 추론을 수행한다

### 요구사항 6: 서버 운영

**사용자 스토리:** 개발자로서, Flask 기반 서버를 안정적으로 운영하고 관리할 수 있어야 합니다.

#### 승인 기준

1. THE Building_Material_Classifier SHALL Flask 프레임워크를 사용하여 웹 서버를 구동한다
2. THE Building_Material_Classifier SHALL 8000번 포트에서 서비스를 제공한다
3. THE Building_Material_Classifier SHALL /api/health 엔드포인트를 통해 서버 상태를 확인할 수 있다
4. THE Building_Material_Classifier SHALL /api/predict 엔드포인트를 통해 이미지 분류를 수행한다
5. THE Building_Material_Classifier SHALL 반응형 디자인을 적용하여 다양한 화면 크기에서 최적화된 경험을 제공한다
