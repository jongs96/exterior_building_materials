"""
Flask 애플리케이션 설정 파일
환경별 설정과 보안 관련 설정을 관리합니다.
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).parent.parent


class Config:
    """기본 설정 클래스"""
    
    # 기본 Flask 설정
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # 파일 업로드 설정
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB 제한
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    
    # 모델 설정
    MODEL_PATH = BASE_DIR / 'building_material_classifier_pytorch.pth'
    MODEL_TYPE = 'pytorch'  # 'pytorch' 또는 'tensorflow'
    
    # 이미지 처리 설정
    IMAGE_SIZE = (224, 224)  # 모델 입력 크기
    
    # 분류 클래스 정의
    CLASS_NAMES = ['brick', 'metal', 'stone', 'stucco', 'wood']
    CLASS_DESCRIPTIONS = {
        'brick': {
            'name': '벽돌/조적',
            'description': '점토를 구워 만든 건축 자재로, 내구성이 뛰어나고 단열 효과가 좋습니다. 전통적이면서도 현대적인 외관을 연출할 수 있습니다.',
            'characteristics': ['높은 내구성', '우수한 단열성', '방화성', '다양한 색상과 질감']
        },
        'metal': {
            'name': '금속 패널',
            'description': '알루미늄, 징크, 스테인리스 스틸 등으로 제작된 외장재입니다. 현대적이고 세련된 외관과 함께 우수한 내구성을 제공합니다.',
            'characteristics': ['현대적 디자인', '가벼운 무게', '내부식성', '다양한 마감 처리']
        },
        'stone': {
            'name': '석재',
            'description': '화강석, 대리석, 사암 등 천연석 또는 인조석으로 제작된 외장재입니다. 고급스럽고 자연스러운 외관을 연출합니다.',
            'characteristics': ['고급스러운 외관', '뛰어난 내구성', '자연친화적', '다양한 질감과 색상']
        },
        'stucco': {
            'name': '스타코/미장',
            'description': '시멘트, 석회, 모래 등을 혼합한 미장재로 마감한 외벽입니다. 부드럽고 균일한 표면과 다양한 질감 표현이 가능합니다.',
            'characteristics': ['매끄러운 표면', '균일한 마감', '다양한 질감', '경제적']
        },
        'wood': {
            'name': '목재 사이딩',
            'description': '천연 목재 또는 목질 복합재로 제작된 외장재입니다. 따뜻하고 자연스러운 느낌을 주며 친환경적입니다.',
            'characteristics': ['자연스러운 외관', '친환경적', '단열 효과', '따뜻한 느낌']
        }
    }
    
    # 로깅 설정
    LOG_LEVEL = 'INFO'
    LOG_FILE = BASE_DIR / 'app.log'
    
    # CORS 설정
    CORS_ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000']


class DevelopmentConfig(Config):
    """개발 환경 설정"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """프로덕션 환경 설정"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """테스트 환경 설정"""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False


# 환경별 설정 매핑
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}